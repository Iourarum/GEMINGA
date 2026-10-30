//! Search that skips what it can prove it does not need.
//!
//! Two mechanisms, same idea as an index-first grep: decide from cheap metadata which parts of
//! the dataset *could* match, fetch only those, verify there.
//!
//! * **Parquet** — every row group's footer carries per-column min/max. A range predicate rules
//!   out whole row groups without a single data page leaving the server. On a remote file that
//!   is the difference between one range request and a download.
//! * **Sequences** — Aho-Corasick matches every pattern in one pass over the bytes, so searching
//!   for 300 adapters costs about what searching for one costs.

use crate::error::{GemingaError, Result};
use aho_corasick::{AhoCorasick, MatchKind};
use arrow::array::{Array, Float64Array, RecordBatch};
use arrow::compute::{and_kleene, cast, filter_record_batch, kernels::cmp};
use arrow::datatypes::DataType;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics;

/// Keep only the rows of `batch` whose `column` lies in `[lo, hi]`.
///
/// Pruning decides which row groups to *fetch*; this decides which rows to *return*. Nulls
/// never match. Any numeric or boolean column is compared in f64, which is exact for every
/// integer a Parquet statistic can express (i32, i64 up to 2^53) and for the floats themselves.
pub fn filter_range(batch: &RecordBatch, column: &str, lo: f64, hi: f64) -> Result<RecordBatch> {
    let idx = batch.schema().index_of(column).map_err(|_| {
        GemingaError::Invalid(format!("column {column:?} not in the projected batch"))
    })?;
    let col = batch.column(idx);
    let as_f64 = match col.data_type() {
        DataType::Float64 => col.clone(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            return Err(GemingaError::Invalid(format!(
                "column {column:?} is {:?}; range search needs a numeric column",
                col.data_type()
            )))
        }
        _ => cast(col.as_ref(), &DataType::Float64)?,
    };
    let lo_s = Float64Array::new_scalar(lo);
    let hi_s = Float64Array::new_scalar(hi);
    let ge = cmp::gt_eq(&as_f64, &lo_s)?;
    let le = cmp::lt_eq(&as_f64, &hi_s)?;
    let mask = and_kleene(&ge, &le)?;
    if mask.null_count() == 0 && mask.true_count() == mask.len() {
        return Ok(batch.clone());
    }
    Ok(filter_record_batch(batch, &mask)?)
}

/// What a predicate would cost, decided entirely from footers.
#[derive(Debug, Clone)]
pub struct Pruning {
    pub row_groups: Vec<usize>,
    pub total_row_groups: usize,
    pub candidate_rows: i64,
    pub total_rows: i64,
    /// Compressed bytes that must be fetched for the selected columns of the kept row groups.
    pub bytes_to_fetch: i64,
    /// Compressed bytes of those columns across the whole file.
    pub bytes_total: i64,
    /// Row groups where the column carried no statistics and had to be kept blind.
    pub kept_without_stats: usize,
}

impl Pruning {
    #[allow(dead_code)]
    pub fn selectivity(&self) -> f64 {
        if self.bytes_total > 0 {
            self.bytes_to_fetch as f64 / self.bytes_total as f64
        } else {
            1.0
        }
    }
}

fn stat_range(s: &Statistics) -> Option<(f64, f64)> {
    Some(match s {
        Statistics::Int32(v) => (*v.min_opt()? as f64, *v.max_opt()? as f64),
        Statistics::Int64(v) => (*v.min_opt()? as f64, *v.max_opt()? as f64),
        Statistics::Float(v) => (*v.min_opt()? as f64, *v.max_opt()? as f64),
        Statistics::Double(v) => (*v.min_opt()?, *v.max_opt()?),
        Statistics::Boolean(v) => (*v.min_opt()? as u8 as f64, *v.max_opt()? as u8 as f64),
        _ => return None,
    })
}

/// Row groups that could contain `column` values within `[lo, hi]`.
/// Row groups whose statistics are missing are kept — pruning must never lose a row.
pub fn prune(
    meta: &ParquetMetaData,
    column: &str,
    lo: f64,
    hi: f64,
    projection: Option<&Vec<String>>,
) -> Result<Pruning> {
    let mut found_column = false;
    let mut keep = Vec::new();
    let mut candidate_rows = 0i64;
    let mut total_rows = 0i64;
    let mut bytes_to_fetch = 0i64;
    let mut bytes_total = 0i64;
    let mut blind = 0usize;

    let wanted = |p: &str| match projection {
        None => true,
        Some(cols) => cols
            .iter()
            .any(|w| p == *w || p.starts_with(&format!("{w}."))),
    };

    for (i, rg) in meta.row_groups().iter().enumerate() {
        total_rows += rg.num_rows();
        let rg_bytes: i64 = rg
            .columns()
            .iter()
            .filter(|c| wanted(&c.column_path().string()))
            .map(|c| c.compressed_size())
            .sum();
        bytes_total += rg_bytes;

        let mut hit = None;
        for c in rg.columns() {
            if c.column_path().string() == column {
                found_column = true;
                hit = Some(c);
                break;
            }
        }
        let overlaps = match hit.and_then(|c| c.statistics()).and_then(stat_range) {
            Some((mn, mx)) => !(mx < lo || mn > hi),
            None => {
                if hit.is_some() {
                    blind += 1;
                }
                true // no statistics: cannot rule it out
            }
        };
        if overlaps {
            keep.push(i);
            candidate_rows += rg.num_rows();
            bytes_to_fetch += rg_bytes;
        }
    }
    if !found_column {
        let avail: Vec<String> = meta
            .row_groups()
            .first()
            .map(|rg| {
                rg.columns()
                    .iter()
                    .map(|c| c.column_path().string())
                    .collect()
            })
            .unwrap_or_default();
        return Err(GemingaError::Invalid(format!(
            "no column {column:?} in this file; available: {avail:?}"
        )));
    }
    Ok(Pruning {
        row_groups: keep,
        total_row_groups: meta.num_row_groups(),
        candidate_rows,
        total_rows,
        bytes_to_fetch,
        bytes_total,
        kept_without_stats: blind,
    })
}

/// One hit: which record, which pattern, where in the sequence.
#[derive(Debug, Clone)]
pub struct SeqHit {
    pub record: u64,
    pub pattern: usize,
    pub offset: u64,
    pub id: String,
}

/// Multi-pattern sequence matcher. Optionally also matches the reverse complement of each
/// pattern, which is what you want for adapters and primers on unstranded libraries.
pub struct SeqSearcher {
    ac: AhoCorasick,
    /// Maps an internal pattern id back to the user's pattern index.
    owner: Vec<usize>,
    #[allow(dead_code)]
    pub n_patterns: usize,
}

pub fn revcomp(s: &str) -> String {
    s.bytes()
        .rev()
        .map(|b| match b | 0x20 {
            b'a' => 'T',
            b't' | b'u' => 'A',
            b'g' => 'C',
            b'c' => 'G',
            b'r' => 'Y',
            b'y' => 'R',
            b'k' => 'M',
            b'm' => 'K',
            _ => 'N',
        })
        .collect()
}

impl SeqSearcher {
    pub fn new(patterns: &[String], both_strands: bool) -> Result<Self> {
        if patterns.is_empty() {
            return Err(GemingaError::Invalid("no patterns given".into()));
        }
        let mut pats: Vec<String> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        for (i, p) in patterns.iter().enumerate() {
            if p.is_empty() {
                return Err(GemingaError::Invalid("empty pattern".into()));
            }
            pats.push(p.to_ascii_uppercase());
            owner.push(i);
            if both_strands {
                let rc = revcomp(p);
                if rc != p.to_ascii_uppercase() {
                    pats.push(rc);
                    owner.push(i);
                }
            }
        }
        let ac = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostFirst)
            .ascii_case_insensitive(true)
            .build(&pats)
            .map_err(|e| GemingaError::Invalid(format!("cannot build matcher: {e}")))?;
        Ok(Self {
            ac,
            owner,
            n_patterns: patterns.len(),
        })
    }

    /// All hits in one sequence.
    pub fn find(&self, seq: &[u8], record: u64, id: &str, out: &mut Vec<SeqHit>, first_only: bool) {
        for m in self.ac.find_iter(seq) {
            out.push(SeqHit {
                record,
                pattern: self.owner[m.pattern().as_usize()],
                offset: m.start() as u64,
                id: id.to_string(),
            });
            if first_only {
                break;
            }
        }
    }
}

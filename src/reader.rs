//! The streaming core. Everything here is generic over `object_store::ObjectStore`, so the
//! exact same code path serves HTTP range requests (Hugging Face Hub, any CDN) and local files.
//!
//! Key idea: a Parquet file's footer describes every row group and column chunk, with byte
//! offsets and per-chunk statistics. Reading the footer costs one or two small range requests.
//! After that we can answer "what is in this dataset?" (schema, row counts, min/max/nulls per
//! column) without touching the data, and fetch exactly the row groups / columns we want.

use crate::budget::{BudgetInner, Lease};
use crate::error::{GemingaError, Result};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt};
use futures::FutureExt;
use object_store::path::Path;
use object_store::{GetOptions, GetRange, ObjectStore, ObjectStoreExt};
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::{AsyncFileReader, MetadataSuffixFetch, ParquetRecordBatchStream};
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::errors::{ParquetError, Result as PqResult};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::statistics::Statistics;
use std::collections::BTreeMap;
use std::ops::Range;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// How much actually crossed the wire. Shared by every file of a dataset.
#[derive(Debug, Default)]
pub struct Telemetry {
    pub bytes: AtomicU64,
    pub range_fetches: AtomicU64,
}

impl Telemetry {
    fn add(&self, bytes: usize, fetches: u64) {
        self.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        self.range_fetches.fetch_add(fetches, Ordering::Relaxed);
    }
    pub fn reset(&self) {
        self.bytes.store(0, Ordering::Relaxed);
        self.range_fetches.store(0, Ordering::Relaxed);
    }
}

/// Footer prefetch: one request covers footers up to this size; larger ones cost a second request.
const FOOTER_PREFETCH: usize = 64 * 1024;

/// An `AsyncFileReader` over any `ObjectStore`, issuing exactly the byte ranges Parquet asks for.
/// Adjacent ranges are coalesced by `object_store::get_ranges`; nothing else is read.
#[derive(Clone)]
pub struct RangeReader {
    store: Arc<dyn ObjectStore>,
    path: Path,
    size: Option<u64>,
    tele: Arc<Telemetry>,
    discovered_size: Arc<AtomicU64>,
}

fn to_pq(e: object_store::Error) -> ParquetError {
    ParquetError::External(Box::new(e))
}

impl AsyncFileReader for RangeReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, PqResult<Bytes>> {
        async move {
            let b = self.store.get_range(&self.path, range).await.map_err(to_pq)?;
            self.tele.add(b.len(), 1);
            Ok(b)
        }
        .boxed()
    }

    fn get_byte_ranges(&mut self, ranges: Vec<Range<u64>>) -> BoxFuture<'_, PqResult<Vec<Bytes>>> {
        async move {
            let n = ranges.len() as u64;
            let v = self.store.get_ranges(&self.path, &ranges).await.map_err(to_pq)?;
            self.tele.add(v.iter().map(|b| b.len()).sum(), n);
            Ok(v)
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, PqResult<Arc<ParquetMetaData>>> {
        async move {
            let reader = ParquetMetaDataReader::new()
                .with_arrow_reader_options(options)
                .with_prefetch_hint(Some(FOOTER_PREFETCH));
            let md = match self.size {
                // Known size (Hub listing / local file): read the tail directly, no HEAD request.
                Some(sz) => reader.load_and_finish(&mut *self, sz).await?,
                // Unknown size: ask for the last N bytes with a suffix range.
                None => reader.load_via_suffix_and_finish(&mut *self).await?,
            };
            Ok(Arc::new(md))
        }
        .boxed()
    }
}

impl MetadataSuffixFetch for &mut RangeReader {
    fn fetch_suffix(&mut self, suffix: usize) -> BoxFuture<'_, PqResult<Bytes>> {
        let options = GetOptions { range: Some(GetRange::Suffix(suffix as u64)), ..Default::default() };
        async move {
            let resp = self.store.get_opts(&self.path, options).await.map_err(to_pq)?;
            self.discovered_size.store(resp.meta.size, Ordering::Relaxed);
            let b = resp.bytes().await.map_err(to_pq)?;
            self.tele.add(b.len(), 1);
            Ok(b)
        }
        .boxed()
    }
}

/// One Parquet file, wherever it lives.
#[derive(Clone)]
pub struct Source {
    pub store: Arc<dyn ObjectStore>,
    pub path: Path,
    /// Known size lets the reader skip a HEAD request and go straight to the footer.
    pub size: Option<u64>,
    pub label: String,
    pub tele: Arc<Telemetry>,
    /// Filled in after the first suffix read when `size` was unknown (0 = still unknown).
    pub discovered_size: Arc<AtomicU64>,
}

/// Average uncompressed bytes per row, from the footer — used to size a batch to a byte budget.
pub fn bytes_per_row(meta: &ParquetMetaData) -> f64 {
    let rows: i64 = meta.row_groups().iter().map(|rg| rg.num_rows()).sum();
    let bytes: i64 = meta.row_groups().iter().map(|rg| rg.total_byte_size()).sum();
    if rows > 0 { bytes as f64 / rows as f64 } else { 1.0 }
}

/// Rows that fit in `chunk_bytes`, given the file's own width. Clamped to something sane.
pub fn rows_for_bytes(meta: &ParquetMetaData, chunk_bytes: u64, columns: Option<&Vec<String>>) -> usize {
    let mut per_row = bytes_per_row(meta);
    // Projection shrinks the row: scale by the fraction of columns kept, by uncompressed size.
    if let Some(cols) = columns {
        let total: i64 = meta.row_groups().iter().flat_map(|rg| rg.columns()).map(|c| c.uncompressed_size()).sum();
        let kept: i64 = meta
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns())
            .filter(|c| {
                let p = c.column_path().string();
                cols.iter().any(|w| p == *w || p.starts_with(&format!("{w}.")))
            })
            .map(|c| c.uncompressed_size())
            .sum();
        if total > 0 && kept > 0 {
            per_row *= kept as f64 / total as f64;
        }
    }
    ((chunk_bytes as f64 / per_row.max(1.0)) as usize).clamp(256, 4_194_304)
}

#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    pub columns: Option<Vec<String>>,
    pub row_groups: Option<Vec<usize>>,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// When set, batch_size is derived from this many bytes instead of being fixed.
    pub chunk_bytes: Option<u64>,
}

pub type BoxedBatchStream = Pin<Box<ParquetRecordBatchStream<RangeReader>>>;

impl Source {
    /// Size given up front, or learned from the first suffix read.
    pub fn known_size(&self) -> Option<u64> {
        self.size.or_else(|| {
            let d = self.discovered_size.load(Ordering::Relaxed);
            (d > 0).then_some(d)
        })
    }

    fn reader(&self) -> RangeReader {
        RangeReader {
            store: self.store.clone(),
            path: self.path.clone(),
            size: self.size.or(self.known_size()),
            tele: self.tele.clone(),
            discovered_size: self.discovered_size.clone(),
        }
    }

    /// Footer only. No data pages are read.
    pub async fn metadata(&self) -> Result<ArrowReaderMetadata> {
        let mut r = self.reader();
        Ok(ArrowReaderMetadata::load_async(&mut r, ArrowReaderOptions::new()).await?)
    }

    /// Build a lazy stream of record batches: only the selected row groups and columns are fetched.
    pub async fn stream(
        &self,
        meta: ArrowReaderMetadata,
        opts: &StreamOptions,
    ) -> Result<BoxedBatchStream> {
        let mut b = ParquetRecordBatchStreamBuilder::new_with_metadata(self.reader(), meta);
        if let Some(cols) = &opts.columns {
            validate_columns(b.schema(), cols)?;
            let mask = ProjectionMask::columns(b.parquet_schema(), cols.iter().map(|s| s.as_str()));
            b = b.with_projection(mask);
        }
        if let Some(rgs) = &opts.row_groups {
            let n = b.metadata().num_row_groups();
            if let Some(bad) = rgs.iter().find(|&&i| i >= n) {
                return Err(GemingaError::Invalid(format!(
                    "row group {bad} out of range for {} ({n} row groups)",
                    self.label
                )));
            }
            b = b.with_row_groups(rgs.clone());
        }
        if opts.batch_size > 0 {
            b = b.with_batch_size(opts.batch_size);
        }
        if let Some(off) = opts.offset {
            b = b.with_offset(off);
        }
        if let Some(lim) = opts.limit {
            b = b.with_limit(lim);
        }
        Ok(Box::pin(b.build()?))
    }
}

fn validate_columns(schema: &SchemaRef, cols: &[String]) -> Result<()> {
    let missing: Vec<&String> = cols
        .iter()
        .filter(|c| schema.column_with_name(c.split('.').next().unwrap_or(c)).is_none())
        .collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(GemingaError::Invalid(format!(
            "unknown column(s) {:?}; available: {:?}",
            missing,
            schema.fields().iter().map(|f| f.name().clone()).collect::<Vec<_>>()
        )))
    }
}

/// Load metadata for many files with bounded concurrency (one footer read each).
pub async fn load_all_metadata(
    sources: &[Source],
    concurrency: usize,
) -> Result<Vec<ArrowReaderMetadata>> {
    let results: Vec<Result<(usize, ArrowReaderMetadata)>> = stream::iter(sources.iter().enumerate())
        .map(|(i, s)| async move { s.metadata().await.map(|m| (i, m)) })
        .buffer_unordered(concurrency.max(1))
        .collect()
        .await;
    let mut out: Vec<Option<ArrowReaderMetadata>> = (0..sources.len()).map(|_| None).collect();
    for r in results {
        let (i, m) = r?;
        out[i] = Some(m);
    }
    Ok(out.into_iter().map(|m| m.expect("metadata slot filled")).collect())
}

// ---------------------------------------------------------------------------
// Metadata-only description and statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RowGroupInfo {
    pub index: usize,
    pub num_rows: i64,
    pub compressed_bytes: i64,
    pub uncompressed_bytes: i64,
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub label: String,
    pub size_bytes: Option<u64>,
    pub num_rows: i64,
    pub num_row_groups: usize,
    pub created_by: Option<String>,
    pub key_value_metadata: Vec<(String, Option<String>)>,
    pub row_groups: Vec<RowGroupInfo>,
}

pub fn describe_file(src: &Source, meta: &ParquetMetaData) -> FileInfo {
    let fm = meta.file_metadata();
    FileInfo {
        label: src.label.clone(),
        size_bytes: src.known_size(),
        num_rows: fm.num_rows(),
        num_row_groups: meta.num_row_groups(),
        created_by: fm.created_by().map(|s| s.to_string()),
        key_value_metadata: fm
            .key_value_metadata()
            .map(|kv| kv.iter().map(|k| (k.key.clone(), k.value.clone())).collect())
            .unwrap_or_default(),
        row_groups: meta
            .row_groups()
            .iter()
            .enumerate()
            .map(|(i, rg)| RowGroupInfo {
                index: i,
                num_rows: rg.num_rows(),
                compressed_bytes: rg.compressed_size(),
                uncompressed_bytes: rg.total_byte_size(),
            })
            .collect(),
    }
}

/// A typed min/max accumulated across row groups (and files).
#[derive(Debug, Clone)]
pub enum MinMax {
    Int { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Bool { min: bool, max: bool },
    Text { min: String, max: String },
    Bytes { min_len: usize, max_len: usize },
}

#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub name: String,
    pub physical_type: String,
    pub num_values: i64,
    pub null_count: Option<u64>,
    pub compressed_bytes: i64,
    pub uncompressed_bytes: i64,
    pub min_max: Option<MinMax>,
    /// Number of row-group chunks that carried statistics for this column.
    pub chunks_with_stats: usize,
    pub chunks_total: usize,
}

fn merge_minmax(acc: &mut Option<MinMax>, s: &Statistics) {
    let next = match s {
        Statistics::Boolean(v) => match (v.min_opt(), v.max_opt()) {
            (Some(&mn), Some(&mx)) => MinMax::Bool { min: mn, max: mx },
            _ => return,
        },
        Statistics::Int32(v) => match (v.min_opt(), v.max_opt()) {
            (Some(&mn), Some(&mx)) => MinMax::Int { min: mn as i64, max: mx as i64 },
            _ => return,
        },
        Statistics::Int64(v) => match (v.min_opt(), v.max_opt()) {
            (Some(&mn), Some(&mx)) => MinMax::Int { min: mn, max: mx },
            _ => return,
        },
        Statistics::Int96(_) => return,
        Statistics::Float(v) => match (v.min_opt(), v.max_opt()) {
            (Some(&mn), Some(&mx)) => MinMax::Float { min: mn as f64, max: mx as f64 },
            _ => return,
        },
        Statistics::Double(v) => match (v.min_opt(), v.max_opt()) {
            (Some(&mn), Some(&mx)) => MinMax::Float { min: mn, max: mx },
            _ => return,
        },
        Statistics::ByteArray(v) => match (v.min_opt(), v.max_opt()) {
            (Some(mn), Some(mx)) => match (mn.as_utf8(), mx.as_utf8()) {
                (Ok(a), Ok(b)) => MinMax::Text { min: a.to_string(), max: b.to_string() },
                _ => MinMax::Bytes { min_len: mn.len(), max_len: mx.len() },
            },
            _ => return,
        },
        Statistics::FixedLenByteArray(v) => match (v.min_opt(), v.max_opt()) {
            (Some(mn), Some(mx)) => MinMax::Bytes { min_len: mn.len(), max_len: mx.len() },
            _ => return,
        },
    };
    *acc = Some(match (acc.take(), next) {
        (None, n) => n,
        (Some(MinMax::Int { min, max }), MinMax::Int { min: a, max: b }) => {
            MinMax::Int { min: min.min(a), max: max.max(b) }
        }
        (Some(MinMax::Float { min, max }), MinMax::Float { min: a, max: b }) => {
            MinMax::Float { min: min.min(a), max: max.max(b) }
        }
        (Some(MinMax::Bool { min, max }), MinMax::Bool { min: a, max: b }) => {
            MinMax::Bool { min: min & a, max: max | b }
        }
        (Some(MinMax::Text { min, max }), MinMax::Text { min: a, max: b }) => MinMax::Text {
            min: if a < min { a } else { min },
            max: if b > max { b } else { max },
        },
        (Some(MinMax::Bytes { min_len, max_len }), MinMax::Bytes { min_len: a, max_len: b }) => {
            MinMax::Bytes { min_len: min_len.min(a), max_len: max_len.max(b) }
        }
        (Some(prev), _) => prev, // type mismatch across files: keep what we had
    });
}

/// Accumulate per-column statistics across any number of files, purely from footers.
pub fn column_stats(metas: &[Arc<ParquetMetaData>]) -> Vec<ColumnStats> {
    let mut acc: BTreeMap<String, ColumnStats> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    for meta in metas {
        for rg in meta.row_groups() {
            for col in rg.columns() {
                let name = col.column_path().string();
                let entry = acc.entry(name.clone()).or_insert_with(|| {
                    order.push(name.clone());
                    ColumnStats {
                        name: name.clone(),
                        physical_type: format!("{:?}", col.column_type()),
                        num_values: 0,
                        null_count: None,
                        compressed_bytes: 0,
                        uncompressed_bytes: 0,
                        min_max: None,
                        chunks_with_stats: 0,
                        chunks_total: 0,
                    }
                });
                entry.num_values += col.num_values();
                entry.compressed_bytes += col.compressed_size();
                entry.uncompressed_bytes += col.uncompressed_size();
                entry.chunks_total += 1;
                if let Some(s) = col.statistics() {
                    entry.chunks_with_stats += 1;
                    if let Some(n) = s.null_count_opt() {
                        entry.null_count = Some(entry.null_count.unwrap_or(0) + n);
                    }
                    merge_minmax(&mut entry.min_max, s);
                }
            }
        }
    }
    order.into_iter().filter_map(|n| acc.remove(&n)).collect()
}

// ---------------------------------------------------------------------------
// Sampling: pick row groups spread across the whole dataset
// ---------------------------------------------------------------------------

/// Choose `k` indices out of `total`, evenly spread; with a seed, jitter each pick within its
/// stride so repeated calls with different seeds see different data but the same coverage.
pub fn spread_indices(total: usize, k: usize, seed: Option<u64>) -> Vec<usize> {
    if k == 0 || total == 0 {
        return vec![];
    }
    if k >= total {
        return (0..total).collect();
    }
    let stride = total as f64 / k as f64;
    let mut rng = seed.map(XorShift64::new);
    (0..k)
        .map(|i| {
            let base = (i as f64 * stride) as usize;
            let span = ((((i + 1) as f64) * stride) as usize).saturating_sub(base).max(1);
            let jitter = rng.as_mut().map_or(0, |r| (r.next() as usize) % span);
            (base + jitter).min(total - 1)
        })
        .collect()
}

pub struct XorShift64(u64);
impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

// ---------------------------------------------------------------------------
// Multi-file sequential iteration with a global row limit
// ---------------------------------------------------------------------------

pub struct MultiFileIter {
    pub sources: Vec<Source>,
    pub metas: Vec<Option<ArrowReaderMetadata>>,
    pub opts: StreamOptions,
    pub next_file: usize,
    pub current: Option<BoxedBatchStream>,
    pub remaining: Option<usize>,
    pub rows_yielded: usize,
    pub batches_yielded: usize,
    pub budget: Option<Arc<BudgetInner>>,
    /// Lease for the batch the consumer currently holds; released when the next one is requested.
    pub lease: Lease,
}

impl MultiFileIter {
    pub fn new(sources: Vec<Source>, metas: Vec<Option<ArrowReaderMetadata>>, opts: StreamOptions) -> Self {
        let remaining = opts.limit;
        Self {
            sources, metas, opts, next_file: 0, current: None, remaining,
            rows_yielded: 0, batches_yielded: 0, budget: None, lease: Lease::none(),
        }
    }

    pub fn with_budget(mut self, b: Option<Arc<BudgetInner>>) -> Self {
        self.budget = b;
        self
    }

    pub async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        // The consumer is asking for the next chunk, so it is done with the previous one.
        self.lease.release();
        loop {
            if matches!(self.remaining, Some(0)) {
                return Ok(None);
            }
            if self.current.is_none() {
                if self.next_file >= self.sources.len() {
                    return Ok(None);
                }
                let i = self.next_file;
                self.next_file += 1;
                let meta = match self.metas[i].take() {
                    Some(m) => m,
                    None => self.sources[i].metadata().await?,
                };
                let mut opts = self.opts.clone();
                opts.limit = self.remaining;
                // Size the batch to the byte budget using this file's own row width.
                if let Some(cb) = self.opts.chunk_bytes {
                    opts.batch_size = rows_for_bytes(meta.metadata(), cb, opts.columns.as_ref());
                }
                // A per-file row-group selection only makes sense for single-file datasets.
                if self.sources.len() > 1 {
                    opts.row_groups = None;
                }
                self.current = Some(self.sources[i].stream(meta, &opts).await?);
                // Offset applies to the first file only.
                self.opts.offset = None;
            }
            let stream = self.current.as_mut().expect("stream set above");
            match stream.next().await {
                Some(Ok(batch)) => {
                    if let Some(b) = &self.budget {
                        let sz = crate::fastx::batch_bytes(&batch);
                        b.acquire("ram", sz)?;
                        self.lease = Lease::new(Some(b.clone()), "ram", sz);
                    }
                    let n = batch.num_rows();
                    self.rows_yielded += n;
                    self.batches_yielded += 1;
                    if let Some(r) = self.remaining.as_mut() {
                        *r = r.saturating_sub(n);
                    }
                    if n == 0 {
                        continue;
                    }
                    return Ok(Some(batch));
                }
                Some(Err(e)) => return Err(e.into()),
                None => {
                    self.current = None;
                }
            }
        }
    }
}

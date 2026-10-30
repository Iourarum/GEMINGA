//! Level-of-detail index for sequence files.
//!
//! Zooming needs two things a raw FASTQ does not give you: **summaries** cheap enough to draw a
//! whole 60 GB file at once, and **checkpoints** so you can jump to record 41,000,000 without
//! re-reading everything before it. One pass over the file produces both, and the sidecar is a
//! few hundred kilobytes regardless of how big the file is.
//!
//! Once the index exists, every later question — "what does the middle of this file look like",
//! "give me records 4.1M to 4.2M" — is answered from the index plus one seek, not a rescan.

use crate::error::{GemingaError, Result};
use needletail::parse_fastx_file;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};

/// One level-of-detail bucket: a contiguous run of records, summarised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqBucket {
    pub start_record: u64,
    pub end_record: u64,
    pub n: u64,
    pub bases: u64,
    pub min_len: u32,
    pub max_len: u32,
    pub gc: f64,
    /// Mean Phred score; NaN for FASTA.
    pub mean_q: f64,
    pub n_frac: f64,
    /// Byte offset of the first record of this bucket (plain files only).
    pub byte_offset: u64,
}

impl SeqBucket {
    fn new(start: u64, off: u64) -> Self {
        Self {
            start_record: start,
            end_record: start,
            n: 0,
            bases: 0,
            min_len: u32::MAX,
            max_len: 0,
            gc: 0.0,
            mean_q: 0.0,
            n_frac: 0.0,
            byte_offset: off,
        }
    }
    pub fn mean_len(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.bases as f64 / self.n as f64
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeqIndex {
    pub path: String,
    pub file_bytes: u64,
    pub n_records: u64,
    pub total_bases: u64,
    pub has_quality: bool,
    /// True when byte offsets can be used to seek (plain, uncompressed input).
    pub seekable: bool,
    pub buckets: Vec<SeqBucket>,
    /// Average payload bytes per record — the basis of every cost estimate.
    pub bytes_per_record: f64,
}

impl SeqIndex {
    /// One pass. `n_buckets` sets the coarsest zoom level's resolution.
    pub fn build(path: &str, n_buckets: usize) -> Result<Self> {
        let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let seekable = !path.ends_with(".gz") && !path.ends_with(".bgz") && !path.ends_with(".zst");
        let n_buckets = n_buckets.clamp(16, 100_000);

        // Pass one is unavoidable — we do not know the record count up front. Everything after
        // this is index-only.
        let mut reader = parse_fastx_file(path)
            .map_err(|e| GemingaError::Invalid(format!("cannot open {path}: {e}")))?;

        let mut buckets: Vec<SeqBucket> = Vec::with_capacity(n_buckets);
        let mut cur = SeqBucket::new(0, 0);
        let mut gc_acc = 0u64;
        let mut n_acc = 0u64;
        let mut q_acc = 0f64;
        let mut q_bases = 0u64;
        let mut total_records = 0u64;
        let mut total_bases = 0u64;
        let mut payload = 0u64;
        let mut has_quality = false;

        // Provisional bucket size; grown (by merging pairs) when we overflow n_buckets.
        let mut per_bucket: u64 = 1024;

        while let Some(rec) = reader.next() {
            let rec = rec.map_err(|e| GemingaError::Invalid(format!("{path}: {e}")))?;

            // A full bucket is closed when the *next* record begins, so the new bucket's
            // byte offset is that record's exact start as the parser saw it — not a sum of
            // field lengths, which missed the '@'/'>' and drifted one byte per record.
            if cur.n >= per_bucket {
                finish(&mut cur, gc_acc, n_acc, q_acc, q_bases);
                buckets.push(cur.clone());
                gc_acc = 0;
                n_acc = 0;
                q_acc = 0.0;
                q_bases = 0;
                cur = SeqBucket::new(total_records, rec.position().byte());

                // Keep the sidecar bounded: merge pairs and double the bucket size.
                if buckets.len() >= n_buckets * 2 {
                    buckets = merge_pairs(&buckets);
                    per_bucket *= 2;
                }
            }

            let seq = rec.raw_seq();
            let id = rec.id();
            let len = seq.len() as u32;

            let mut gc = 0u64;
            let mut ns = 0u64;
            for &b in seq {
                match b | 0x20 {
                    b'g' | b'c' => gc += 1,
                    b'n' => ns += 1,
                    _ => {}
                }
            }
            gc_acc += gc;
            n_acc += ns;

            // '@'/'>' + id + '\n' + seq + '\n' (+ "+\n" + qual + '\n' for FASTQ)
            let mut rec_bytes = id.len() + seq.len() + 3;
            if let Some(q) = rec.qual() {
                has_quality = true;
                let s: u64 = q.iter().map(|&c| (c.saturating_sub(33)) as u64).sum();
                q_acc += s as f64;
                q_bases += q.len() as u64;
                rec_bytes += q.len() + 3;
            }
            payload += rec_bytes as u64;

            cur.n += 1;
            cur.bases += len as u64;
            cur.min_len = cur.min_len.min(len);
            cur.max_len = cur.max_len.max(len);
            cur.end_record = total_records;
            total_records += 1;
            total_bases += len as u64;
        }
        if cur.n > 0 {
            finish(&mut cur, gc_acc, n_acc, q_acc, q_bases);
            buckets.push(cur);
        }

        let bytes_per_record = if total_records > 0 {
            payload as f64 / total_records as f64
        } else {
            0.0
        };

        Ok(Self {
            path: path.to_string(),
            file_bytes,
            n_records: total_records,
            total_bases,
            has_quality,
            seekable,
            buckets,
            bytes_per_record,
        })
    }

    /// Buckets re-aggregated to `target` resolution over `[lo, hi)` — this is what "zoom" means.
    /// Purely arithmetic on the index; nothing is read.
    pub fn zoom(&self, lo: u64, hi: u64, target: usize) -> Vec<SeqBucket> {
        let hi = hi.min(self.n_records);
        if lo >= hi || target == 0 {
            return vec![];
        }
        let sel: Vec<&SeqBucket> = self
            .buckets
            .iter()
            .filter(|b| b.end_record >= lo && b.start_record < hi)
            .collect();
        if sel.is_empty() {
            return vec![];
        }
        if sel.len() <= target {
            return sel.into_iter().cloned().collect();
        }
        let group = (sel.len() as f64 / target as f64).ceil() as usize;
        sel.chunks(group)
            .map(|c| {
                let mut out = SeqBucket::new(c[0].start_record, c[0].byte_offset);
                out.end_record = c[c.len() - 1].end_record;
                let mut gcw = 0.0;
                let mut qw = 0.0;
                let mut nw = 0.0;
                for b in c {
                    out.n += b.n;
                    out.bases += b.bases;
                    out.min_len = out.min_len.min(b.min_len);
                    out.max_len = out.max_len.max(b.max_len);
                    gcw += b.gc * b.bases as f64;
                    nw += b.n_frac * b.bases as f64;
                    if b.mean_q.is_finite() {
                        qw += b.mean_q * b.bases as f64;
                    }
                }
                let w = out.bases.max(1) as f64;
                out.gc = gcw / w;
                out.n_frac = nw / w;
                out.mean_q = if self.has_quality { qw / w } else { f64::NAN };
                out
            })
            .collect()
    }

    /// Record index -> byte offset of the nearest checkpoint at or before it.
    pub fn checkpoint_for(&self, record: u64) -> (u64, u64) {
        let mut best = (0u64, 0u64);
        for b in &self.buckets {
            if b.start_record <= record {
                best = (b.start_record, b.byte_offset);
            } else {
                break;
            }
        }
        best
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer(&mut w, self)?;
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&s)?)
    }
}

fn finish(b: &mut SeqBucket, gc: u64, ns: u64, q: f64, qb: u64) {
    let bases = b.bases.max(1) as f64;
    b.gc = gc as f64 / bases;
    b.n_frac = ns as f64 / bases;
    b.mean_q = if qb > 0 { q / qb as f64 } else { f64::NAN };
    if b.min_len == u32::MAX {
        b.min_len = 0;
    }
}

fn merge_pairs(src: &[SeqBucket]) -> Vec<SeqBucket> {
    src.chunks(2)
        .map(|c| {
            if c.len() == 1 {
                return c[0].clone();
            }
            let (a, b) = (&c[0], &c[1]);
            let bases = (a.bases + b.bases).max(1) as f64;
            SeqBucket {
                start_record: a.start_record,
                end_record: b.end_record,
                n: a.n + b.n,
                bases: a.bases + b.bases,
                min_len: a.min_len.min(b.min_len),
                max_len: a.max_len.max(b.max_len),
                gc: (a.gc * a.bases as f64 + b.gc * b.bases as f64) / bases,
                n_frac: (a.n_frac * a.bases as f64 + b.n_frac * b.bases as f64) / bases,
                mean_q: if a.mean_q.is_finite() && b.mean_q.is_finite() {
                    (a.mean_q * a.bases as f64 + b.mean_q * b.bases as f64) / bases
                } else {
                    f64::NAN
                },
                byte_offset: a.byte_offset,
            }
        })
        .collect()
}

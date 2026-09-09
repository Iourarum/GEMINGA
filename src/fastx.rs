//! FASTA / FASTQ streaming, chunked by *bytes*, not by record count.
//!
//! Bulk RNA-seq FASTQs are the case where Python parsing genuinely hurts: tens of millions of
//! four-line records, each needing a string allocation. needletail parses them in Rust; we
//! accumulate into Arrow builders until the chunk hits its byte target, then hand one
//! `RecordBatch` across the C Data Interface. gzip is detected and decompressed transparently.
//!
//! The chunk boundary is a whole record — never a truncated read — so a chunk is always a
//! valid sub-FASTQ that can be written back out or fed to a tokenizer as-is.

use crate::error::{GemingaError, Result};
use arrow::array::{ArrayRef, Int32Builder, RecordBatch, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use needletail::parse_fastx_file;
use needletail::parser::FastxReader;
use std::sync::Arc;

pub fn fastx_schema(with_qual: bool) -> SchemaRef {
    let mut f = vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("seq", DataType::Utf8, false),
        Field::new("length", DataType::Int32, false),
    ];
    if with_qual {
        f.push(Field::new("qual", DataType::Utf8, true));
    }
    Arc::new(Schema::new(f))
}

pub struct FastxChunker {
    reader: Box<dyn FastxReader>,
    target_bytes: usize,
    max_records: usize,
    with_qual: bool,
    schema: SchemaRef,
    exhausted: bool,
    pub records_read: u64,
    pub bytes_read: u64,
}

impl FastxChunker {
    /// `target_bytes` is the *payload* size of one chunk (ids + sequences + qualities).
    /// The Arrow batch that comes out is close to this, plus offset arrays.
    pub fn new(path: &str, target_bytes: usize, max_records: usize) -> Result<Self> {
        let mut reader = parse_fastx_file(path)
            .map_err(|e| GemingaError::Invalid(format!("cannot open {path} as FASTA/FASTQ: {e}")))?;
        // Peek one record to learn whether this file carries quality scores.
        let with_qual = match reader.next() {
            Some(Ok(rec)) => rec.qual().is_some(),
            Some(Err(e)) => return Err(GemingaError::Invalid(format!("{path}: {e}"))),
            None => false,
        };
        // needletail readers cannot rewind, so reopen after the peek.
        let reader = parse_fastx_file(path)
            .map_err(|e| GemingaError::Invalid(format!("cannot reopen {path}: {e}")))?;
        Ok(Self {
            reader,
            target_bytes: target_bytes.max(1),
            max_records: if max_records == 0 { usize::MAX } else { max_records },
            with_qual,
            schema: fastx_schema(with_qual),
            exhausted: false,
            records_read: 0,
            bytes_read: 0,
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    pub fn with_qual(&self) -> bool {
        self.with_qual
    }

    /// Next chunk, or `None` at end of file. Chunks end on a record boundary.
    pub fn next_chunk(&mut self) -> Result<Option<RecordBatch>> {
        if self.exhausted {
            return Ok(None);
        }
        let mut ids = StringBuilder::new();
        let mut seqs = StringBuilder::new();
        let mut lens = Int32Builder::new();
        let mut quals = StringBuilder::new();
        let mut payload = 0usize;
        let mut n = 0usize;

        while payload < self.target_bytes && n < self.max_records {
            match self.reader.next() {
                Some(Ok(rec)) => {
                    let id = rec.id();
                    let seq = rec.raw_seq();
                    ids.append_value(String::from_utf8_lossy(id));
                    seqs.append_value(String::from_utf8_lossy(seq));
                    lens.append_value(seq.len() as i32);
                    let mut qlen = 0;
                    if self.with_qual {
                        match rec.qual() {
                            Some(q) => {
                                qlen = q.len();
                                quals.append_value(String::from_utf8_lossy(q));
                            }
                            None => quals.append_null(),
                        }
                    }
                    payload += id.len() + seq.len() + qlen;
                    n += 1;
                }
                Some(Err(e)) => {
                    return Err(GemingaError::Invalid(format!(
                        "parse error after {} records: {e}",
                        self.records_read
                    )))
                }
                None => {
                    self.exhausted = true;
                    break;
                }
            }
        }

        if n == 0 {
            return Ok(None);
        }
        self.records_read += n as u64;
        self.bytes_read += payload as u64;

        let mut cols: Vec<ArrayRef> = vec![
            Arc::new(ids.finish()),
            Arc::new(seqs.finish()),
            Arc::new(lens.finish()),
        ];
        if self.with_qual {
            cols.push(Arc::new(quals.finish()));
        }
        Ok(Some(RecordBatch::try_new(self.schema.clone(), cols)?))
    }
}

/// Actual resident size of a batch, for budget accounting.
pub fn batch_bytes(b: &RecordBatch) -> u64 {
    b.columns().iter().map(|c| c.get_array_memory_size() as u64).sum()
}

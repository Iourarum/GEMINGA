//! GEMINGA — Memory-Efficient Arrow-Native Dataset Explorer & Reader.
//!
//! Python-facing layer. Every method that touches the network releases the GIL while the
//! tokio runtime does the I/O, and every batch crosses into Python through the Arrow C Data
//! Interface (no copy), so `pyarrow`, `polars`, `pandas`, and `torch` can consume it directly.

mod budget;
mod error;
mod fastx;
mod hub;
mod reader;

use arrow_pyarrow::ToPyArrow;
use error::{GemingaError, Result};
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList};
use budget::{human, parse_size, BudgetInner, Lease, Tier};
use fastx::{batch_bytes, FastxChunker};
use reader::{
    bytes_per_row, rows_for_bytes, column_stats, describe_file, load_all_metadata, spread_indices, ColumnStats, FileInfo, MinMax,
    MultiFileIter, Source, StreamOptions,
};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;

const DEFAULT_BATCH: usize = 65_536;
const META_CONCURRENCY: usize = 8;

fn runtime() -> Result<Arc<Runtime>> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("geminga-io")
        .enable_all()
        .build()?;
    Ok(Arc::new(rt))
}

/// A dataset made of one or more Parquet files, remote or local, read lazily.
#[pyclass(module = "geminga._core")]
pub struct Dataset {
    rt: Arc<Runtime>,
    sources: Vec<Source>,
    origin: String,
    partial: bool,
    meta_cache: Mutex<Option<Vec<ArrowReaderMetadata>>>,
    budget: Mutex<Option<Arc<BudgetInner>>>,
}

impl Dataset {
    fn new(rt: Arc<Runtime>, sources: Vec<Source>, origin: String, partial: bool) -> Self {
        Self { rt, sources, origin, partial, meta_cache: Mutex::new(None), budget: Mutex::new(None) }
    }

    /// Footer metadata for every file, loaded once (concurrently) and cached.
    fn metas(&self, py: Python<'_>) -> Result<Vec<ArrowReaderMetadata>> {
        if let Some(m) = self.meta_cache.lock().unwrap().as_ref() {
            return Ok(m.clone());
        }
        let sources = &self.sources;
        let rt = self.rt.clone();
        let metas = py.detach(|| rt.block_on(load_all_metadata(sources, META_CONCURRENCY)))?;
        *self.meta_cache.lock().unwrap() = Some(metas.clone());
        Ok(metas)
    }

    fn iter_with(&self, py: Python<'_>, opts: StreamOptions, only_files: Option<Vec<usize>>) -> Result<BatchIterator> {
        let metas = self.metas(py)?;
        let (sources, metas): (Vec<Source>, Vec<Option<ArrowReaderMetadata>>) = match only_files {
            Some(idx) => {
                if let Some(bad) = idx.iter().find(|&&i| i >= self.sources.len()) {
                    return Err(GemingaError::Invalid(format!("file index {bad} out of range")));
                }
                idx.into_iter()
                    .map(|i| (self.sources[i].clone(), Some(metas[i].clone())))
                    .unzip()
            }
            None => (self.sources.clone(), metas.into_iter().map(Some).collect()),
        };
        let b = self.budget.lock().unwrap().clone();
        Ok(BatchIterator {
            rt: self.rt.clone(),
            inner: Mutex::new(MultiFileIter::new(sources, metas, opts).with_budget(b)),
        })
    }
}

#[pymethods]
impl Dataset {
    /// Resolve a Hugging Face dataset (e.g. "HuggingFaceTB/smoltalk2") to its Parquet files on
    /// the `refs/convert/parquet` branch via the datasets-server, without downloading anything.
    #[staticmethod]
    #[pyo3(signature = (dataset, config=None, split=None, token=None))]
    fn from_hub(
        py: Python<'_>,
        dataset: String,
        config: Option<String>,
        split: Option<String>,
        token: Option<String>,
    ) -> PyResult<Self> {
        let rt = runtime()?;
        let (files, partial) = {
            let rt = rt.clone();
            let (d, c, s, t) = (dataset.clone(), config.clone(), split.clone(), token.clone());
            py.detach(move || {
                rt.block_on(hub::list_parquet(&d, c.as_deref(), s.as_deref(), t.as_deref()))
            })?
        };
        let mut cache = hub::StoreCache::new(token);
        let mut sources = Vec::with_capacity(files.len());
        for f in &files {
            let label = format!("{}/{}/{}", f.config, f.split, f.filename);
            let size = if f.size > 0 { Some(f.size) } else { None };
            sources.push(cache.source_for_url(&f.url, size, label)?);
        }
        let origin = format!(
            "hf://datasets/{dataset}{}{}",
            config.map(|c| format!(" config={c}")).unwrap_or_default(),
            split.map(|s| format!(" split={s}")).unwrap_or_default()
        );
        Ok(Self::new(rt, sources, origin, partial))
    }

    /// Any HTTP(S) URLs pointing at Parquet files (the server must support Range requests).
    #[staticmethod]
    #[pyo3(signature = (urls, token=None, sizes=None))]
    fn from_urls(urls: Vec<String>, token: Option<String>, sizes: Option<Vec<u64>>) -> PyResult<Self> {
        if urls.is_empty() {
            return Err(GemingaError::Invalid("no urls given".into()).into());
        }
        if let Some(s) = &sizes {
            if s.len() != urls.len() {
                return Err(GemingaError::Invalid("sizes must match urls".into()).into());
            }
        }
        let rt = runtime()?;
        let mut cache = hub::StoreCache::new(token);
        let mut sources = Vec::with_capacity(urls.len());
        for (i, u) in urls.iter().enumerate() {
            let size = sizes.as_ref().map(|s| s[i]);
            sources.push(cache.source_for_url(u, size, u.clone())?);
        }
        Ok(Self::new(rt, sources, format!("{} url(s)", urls.len()), false))
    }

    /// Local Parquet files — same code path, useful for testing and for already-materialized subsets.
    #[staticmethod]
    fn from_paths(paths: Vec<String>) -> PyResult<Self> {
        if paths.is_empty() {
            return Err(GemingaError::Invalid("no paths given".into()).into());
        }
        let rt = runtime()?;
        let cache = hub::StoreCache::new(None);
        let sources = paths
            .iter()
            .map(|p| cache.source_for_local(p))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::new(rt, sources, format!("{} local file(s)", paths.len()), false))
    }

    /// Build the direct `resolve` URL for a file on the Parquet branch.
    #[staticmethod]
    fn resolve_url(repo: String, config: String, split: String, filename: String) -> String {
        hub::resolve_url(&repo, &config, &split, &filename)
    }

    #[getter]
    fn files(&self) -> Vec<String> {
        self.sources.iter().map(|s| s.label.clone()).collect()
    }

    #[getter]
    fn num_files(&self) -> usize {
        self.sources.len()
    }

    #[getter]
    fn partial(&self) -> bool {
        self.partial
    }

    fn __repr__(&self) -> String {
        format!("Dataset({}, files={})", self.origin, self.sources.len())
    }

    /// Bytes and range fetches that actually crossed the wire for this dataset so far.
    fn transfer_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let t = &self.sources[0].tele;
        d.set_item("bytes", t.bytes.load(std::sync::atomic::Ordering::Relaxed))?;
        d.set_item("range_fetches", t.range_fetches.load(std::sync::atomic::Ordering::Relaxed))?;
        Ok(d)
    }

    fn reset_transfer_stats(&self) {
        self.sources[0].tele.reset();
    }

    /// What is in this dataset — from footers only, no data pages are read.
    fn describe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let metas = self.metas(py)?;
        let infos: Vec<FileInfo> = self
            .sources
            .iter()
            .zip(metas.iter())
            .map(|(s, m)| describe_file(s, m.metadata()))
            .collect();
        let out = PyDict::new(py);
        out.set_item("origin", &self.origin)?;
        out.set_item("partial", self.partial)?;
        out.set_item("num_files", self.sources.len())?;
        out.set_item("total_rows", infos.iter().map(|f| f.num_rows).sum::<i64>())?;
        out.set_item("total_row_groups", infos.iter().map(|f| f.num_row_groups).sum::<usize>())?;
        out.set_item(
            "total_size_bytes",
            infos.iter().filter_map(|f| f.size_bytes).sum::<u64>(),
        )?;
        let schema = metas[0].schema();
        let fields: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| {
                format!(
                    "{}: {}{}",
                    f.name(),
                    f.data_type(),
                    if f.is_nullable() { "" } else { " not null" }
                )
            })
            .collect();
        out.set_item("schema", fields)?;
        let files = PyList::empty(py);
        for info in &infos {
            let d = PyDict::new(py);
            d.set_item("file", &info.label)?;
            d.set_item("size_bytes", info.size_bytes)?;
            d.set_item("num_rows", info.num_rows)?;
            d.set_item("num_row_groups", info.num_row_groups)?;
            d.set_item("created_by", info.created_by.as_deref())?;
            let kv = PyDict::new(py);
            for (k, v) in &info.key_value_metadata {
                kv.set_item(k, v.as_deref())?;
            }
            d.set_item("metadata", kv)?;
            let rgs = PyList::empty(py);
            for rg in &info.row_groups {
                let r = PyDict::new(py);
                r.set_item("index", rg.index)?;
                r.set_item("num_rows", rg.num_rows)?;
                r.set_item("compressed_bytes", rg.compressed_bytes)?;
                r.set_item("uncompressed_bytes", rg.uncompressed_bytes)?;
                rgs.append(r)?;
            }
            d.set_item("row_groups", rgs)?;
            files.append(d)?;
        }
        out.set_item("files", files)?;
        Ok(out)
    }

    /// Arrow schema as a `pyarrow.Schema` (from the first file).
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let metas = self.metas(py)?;
        metas[0].schema().as_ref().to_pyarrow(py)
    }

    /// Per-column statistics aggregated over every row group of every file — still footer-only.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let metas = self.metas(py)?;
        let pm: Vec<_> = metas.iter().map(|m| m.metadata().clone()).collect();
        let stats: Vec<ColumnStats> = column_stats(&pm);
        let out = PyList::empty(py);
        for s in stats {
            let d = PyDict::new(py);
            d.set_item("column", &s.name)?;
            d.set_item("physical_type", &s.physical_type)?;
            d.set_item("num_values", s.num_values)?;
            d.set_item("null_count", s.null_count)?;
            d.set_item("compressed_bytes", s.compressed_bytes)?;
            d.set_item("uncompressed_bytes", s.uncompressed_bytes)?;
            d.set_item("chunks_with_stats", s.chunks_with_stats)?;
            d.set_item("chunks_total", s.chunks_total)?;
            match &s.min_max {
                Some(MinMax::Int { min, max }) => {
                    d.set_item("min", *min)?;
                    d.set_item("max", *max)?;
                }
                Some(MinMax::Float { min, max }) => {
                    d.set_item("min", *min)?;
                    d.set_item("max", *max)?;
                }
                Some(MinMax::Bool { min, max }) => {
                    d.set_item("min", *min)?;
                    d.set_item("max", *max)?;
                }
                Some(MinMax::Text { min, max }) => {
                    d.set_item("min", min)?;
                    d.set_item("max", max)?;
                }
                Some(MinMax::Bytes { min_len, max_len }) => {
                    d.set_item("min_len", *min_len)?;
                    d.set_item("max_len", *max_len)?;
                }
                None => {
                    d.set_item("min", py.None())?;
                    d.set_item("max", py.None())?;
                }
            }
            out.append(d)?;
        }
        Ok(out)
    }

    /// Lazy iterator of `pyarrow.RecordBatch`; only the selected columns and row groups are fetched.
    #[pyo3(signature = (columns=None, batch_size=DEFAULT_BATCH, limit=None, offset=None, row_groups=None, files=None, chunk_bytes=None))]
    fn stream(
        &self,
        py: Python<'_>,
        columns: Option<Vec<String>>,
        batch_size: usize,
        limit: Option<usize>,
        offset: Option<usize>,
        row_groups: Option<Vec<usize>>,
        files: Option<Vec<usize>>,
        chunk_bytes: Option<&str>,
    ) -> PyResult<BatchIterator> {
        let cb = match chunk_bytes {
            Some(s) => Some(parse_size(s)?),
            None => self
                .budget
                .lock()
                .unwrap()
                .as_ref()
                // Default: a quarter of the RAM tier, so a few chunks can be in flight.
                .map(|b| (b.ram.capacity / 4).max(1 << 20)),
        };
        let opts = StreamOptions { columns, row_groups, batch_size, limit, offset, chunk_bytes: cb };
        Ok(self.iter_with(py, opts, files)?)
    }

    /// Attach a Budget. Every batch this dataset produces is then accounted against it,
    /// and batch sizes are derived from it unless you pass an explicit chunk_bytes.
    fn set_budget(&self, budget: Option<PyRef<'_, Budget>>) {
        *self.budget.lock().unwrap() = budget.map(|b| b.inner.clone());
    }

    /// What streaming this dataset would cost, before reading any data: chunk count and the
    /// resident bytes per chunk, for a given budget or explicit chunk size.
    #[pyo3(signature = (chunk_bytes=None, columns=None))]
    fn plan<'py>(&self, py: Python<'py>, chunk_bytes: Option<&str>, columns: Option<Vec<String>>) -> PyResult<Bound<'py, PyDict>> {
        let metas = self.metas(py)?;
        let cb = match chunk_bytes {
            Some(s) => parse_size(s)?,
            None => self.budget.lock().unwrap().as_ref().map(|b| (b.ram.capacity / 4).max(1 << 20)).unwrap_or(64 << 20),
        };
        let total_rows: i64 = metas.iter().map(|m| m.metadata().file_metadata().num_rows()).sum();
        let rows = rows_for_bytes(metas[0].metadata(), cb, columns.as_ref());
        let per_row = bytes_per_row(metas[0].metadata());
        let d = PyDict::new(py);
        d.set_item("chunk_bytes", cb)?;
        d.set_item("chunk_bytes_human", human(cb))?;
        d.set_item("rows_per_chunk", rows)?;
        d.set_item("bytes_per_row", per_row)?;
        d.set_item("total_rows", total_rows)?;
        d.set_item("chunks", (total_rows as f64 / rows as f64).ceil() as u64)?;
        d.set_item("uncompressed_total", (total_rows as f64 * per_row) as u64)?;
        d.set_item("uncompressed_total_human", human((total_rows as f64 * per_row) as u64))?;
        Ok(d)
    }

    /// A representative sample: `n_row_groups` row groups spread evenly across all files
    /// (jittered by `seed` if given), returned as a list of `pyarrow.RecordBatch`.
    /// This is the fast path for "show me this dataset in high detail" — a handful of range
    /// requests instead of a download.
    #[pyo3(signature = (n_row_groups=8, seed=None, columns=None, batch_size=DEFAULT_BATCH))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n_row_groups: usize,
        seed: Option<u64>,
        columns: Option<Vec<String>>,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let metas = self.metas(py)?;
        let mut all: Vec<(usize, usize)> = Vec::new();
        for (fi, m) in metas.iter().enumerate() {
            for rg in 0..m.metadata().num_row_groups() {
                all.push((fi, rg));
            }
        }
        let picks = spread_indices(all.len(), n_row_groups, seed);
        let mut by_file: Vec<Vec<usize>> = vec![Vec::new(); self.sources.len()];
        for p in picks {
            let (fi, rg) = all[p];
            by_file[fi].push(rg);
        }
        let out = PyList::empty(py);
        for (fi, rgs) in by_file.into_iter().enumerate() {
            if rgs.is_empty() {
                continue;
            }
            let opts = StreamOptions {
                columns: columns.clone(),
                row_groups: Some(rgs),
                batch_size,
                limit: None,
                offset: None,
                chunk_bytes: None,
            };
            let mut it = MultiFileIter::new(
                vec![self.sources[fi].clone()],
                vec![Some(metas[fi].clone())],
                opts,
            );
            let rt = self.rt.clone();
            loop {
                let next = py.detach(|| rt.block_on(it.next_batch()))?;
                match next {
                    Some(b) => out.append(b.to_pyarrow(py)?)?,
                    None => break,
                }
            }
        }
        Ok(out)
    }

    /// Stream a subset to a local Parquet file (e.g. the slice you want in a training set),
    /// never holding more than one batch in memory.
    #[pyo3(signature = (path, columns=None, limit=None, row_groups=None, batch_size=DEFAULT_BATCH, files=None))]
    fn materialize<'py>(
        &self,
        py: Python<'py>,
        path: String,
        columns: Option<Vec<String>>,
        limit: Option<usize>,
        row_groups: Option<Vec<usize>>,
        batch_size: usize,
        files: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cb = self.budget.lock().unwrap().as_ref().map(|b| (b.ram.capacity / 4).max(1 << 20));
        let opts = StreamOptions { columns, row_groups, batch_size, limit, offset: None, chunk_bytes: cb };
        let it = self.iter_with(py, opts, files)?;
        let rt = self.rt.clone();
        let mut guard = it.inner.lock().map_err(|_| PyRuntimeError::new_err("iterator poisoned"))?;
        let state: &mut MultiFileIter = &mut guard;
        let path_c = path.clone();
        let (rows, batches) = py.detach(move || -> Result<(usize, usize)> {
            let mut writer: Option<ArrowWriter<std::fs::File>> = None;
            let mut rows = 0usize;
            let mut batches = 0usize;
            while let Some(batch) = rt.block_on(state.next_batch())? {
                if writer.is_none() {
                    let file = std::fs::File::create(&path_c)?;
                    let props = WriterProperties::builder()
                        .set_compression(Compression::SNAPPY)
                        .build();
                    writer = Some(ArrowWriter::try_new(file, batch.schema(), Some(props))?);
                }
                writer.as_mut().unwrap().write(&batch)?;
                rows += batch.num_rows();
                batches += 1;
            }
            if let Some(w) = writer {
                w.close()?;
            }
            Ok((rows, batches))
        })?;
        let d = PyDict::new(py);
        d.set_item("path", path)?;
        d.set_item("rows", rows)?;
        d.set_item("batches", batches)?;
        Ok(d)
    }
}

/// Iterator over `pyarrow.RecordBatch` objects; each `next()` fetches at most one batch.
#[pyclass(module = "geminga._core")]
pub struct BatchIterator {
    rt: Arc<Runtime>,
    inner: Mutex<MultiFileIter>,
}

#[pymethods]
impl BatchIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(slf: PyRef<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        let rt = slf.rt.clone();
        let mut guard = slf.inner.lock().map_err(|_| PyRuntimeError::new_err("iterator poisoned"))?;
        let state: &mut MultiFileIter = &mut guard;
        let next = py.detach(|| rt.block_on(state.next_batch()))?;
        match next {
            Some(b) => Ok(Some(b.to_pyarrow(py)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn rows_yielded(&self) -> usize {
        self.inner.lock().map(|g| g.rows_yielded).unwrap_or(0)
    }

    #[getter]
    fn batches_yielded(&self) -> usize {
        self.inner.lock().map(|g| g.batches_yielded).unwrap_or(0)
    }
}


// ---------------------------------------------------------------------------
// Budget: the governor, and a lease you can hold from Python
// ---------------------------------------------------------------------------

/// A memory budget across three tiers. Attach it to readers, or acquire from it by hand
/// around your own allocations (model activations, mask buffers, a decoded image).
///
///     b = geminga.Budget(ram="6GB", vram="4GB", spill="50GB")
///     with b.acquire("vram", "1.2GB"):
///         ...                      # move a batch to the GPU
#[pyclass(module = "geminga._core")]
pub struct Budget {
    inner: Arc<BudgetInner>,
}

fn tier_dict<'py>(py: Python<'py>, t: &Tier) -> PyResult<Bound<'py, PyDict>> {
    let s = t.stats();
    let d = PyDict::new(py);
    d.set_item("capacity", s.capacity)?;
    d.set_item("capacity_human", human(s.capacity))?;
    d.set_item("used", s.used)?;
    d.set_item("used_human", human(s.used))?;
    d.set_item("peak", s.peak)?;
    d.set_item("peak_human", human(s.peak))?;
    d.set_item("grants", s.grants)?;
    d.set_item("waits", s.waits)?;
    d.set_item("wait_millis", s.wait_millis)?;
    d.set_item("denials", s.denials)?;
    Ok(d)
}

#[pymethods]
impl Budget {
    #[new]
    #[pyo3(signature = (ram="4GB", vram="0", spill="0", timeout_s=120.0))]
    fn new(ram: &str, vram: &str, spill: &str, timeout_s: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(BudgetInner {
                ram: Tier::new("ram", parse_size(ram)?),
                vram: Tier::new("vram", parse_size(vram)?),
                spill: Tier::new("spill", parse_size(spill)?),
                timeout: Duration::from_secs_f64(timeout_s.max(0.0)),
            }),
        })
    }

    /// Reserve bytes in a tier. Returns a lease usable as a context manager.
    /// Blocks if the tier is full; raises MemoryError if the request can never fit.
    fn acquire(&self, tier: &str, size: &str) -> PyResult<PyLease> {
        let n = parse_size(size)?;
        self.inner.acquire(tier, n)?;
        Ok(PyLease { lease: Mutex::new(Lease::new(Some(self.inner.clone()), tier, n)) })
    }

    /// Bytes currently free in a tier.
    fn available(&self, tier: &str) -> PyResult<u64> {
        Ok(self.inner.tier(tier)?.available())
    }

    /// Per-tier capacity, current use, peak, grants, waits and denials.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("ram", tier_dict(py, &self.inner.ram)?)?;
        d.set_item("vram", tier_dict(py, &self.inner.vram)?)?;
        d.set_item("spill", tier_dict(py, &self.inner.spill)?)?;
        Ok(d)
    }

    fn reset_peak(&self) {
        self.inner.ram.reset_peak();
        self.inner.vram.reset_peak();
        self.inner.spill.reset_peak();
    }

    fn __repr__(&self) -> String {
        let r = self.inner.ram.stats();
        let v = self.inner.vram.stats();
        format!(
            "Budget(ram={}/{}, vram={}/{}, peak_ram={})",
            human(r.used), human(r.capacity), human(v.used), human(v.capacity), human(r.peak)
        )
    }
}

/// A reservation. Released on __exit__, on release(), or when garbage collected.
#[pyclass(module = "geminga._core", name = "Lease")]
pub struct PyLease {
    lease: Mutex<Lease>,
}

#[pymethods]
impl PyLease {
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyAny>) -> bool {
        self.lease.lock().unwrap().release();
        false
    }

    fn release(&self) {
        self.lease.lock().unwrap().release();
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.lease.lock().unwrap().bytes()
    }
}

// ---------------------------------------------------------------------------
// FASTA / FASTQ
// ---------------------------------------------------------------------------

/// Byte-chunked FASTA/FASTQ reader yielding pyarrow.RecordBatch (id, seq, length[, qual]).
/// gzip is handled transparently; chunks always end on a record boundary.
#[pyclass(module = "geminga._core")]
pub struct FastxStream {
    inner: Mutex<FastxChunker>,
    budget: Option<Arc<BudgetInner>>,
    lease: Mutex<Lease>,
}

#[pymethods]
impl FastxStream {
    #[new]
    #[pyo3(signature = (path, chunk_bytes="64MB", max_records=0, budget=None))]
    fn new(path: &str, chunk_bytes: &str, max_records: usize, budget: Option<PyRef<'_, Budget>>) -> PyResult<Self> {
        let target = parse_size(chunk_bytes)? as usize;
        Ok(Self {
            inner: Mutex::new(FastxChunker::new(path, target, max_records)?),
            budget: budget.map(|b| b.inner.clone()),
            lease: Mutex::new(Lease::none()),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(slf: PyRef<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        slf.lease.lock().unwrap().release();
        let batch = {
            let mut g = slf.inner.lock().map_err(|_| PyRuntimeError::new_err("reader poisoned"))?;
            let state: &mut FastxChunker = &mut g;
            py.detach(|| state.next_chunk())?
        };
        match batch {
            Some(b) => {
                if let Some(bud) = &slf.budget {
                    let sz = batch_bytes(&b);
                    bud.acquire("ram", sz)?;
                    *slf.lease.lock().unwrap() = Lease::new(Some(bud.clone()), "ram", sz);
                }
                Ok(Some(b.to_pyarrow(py)?))
            }
            None => Ok(None),
        }
    }

    #[getter]
    fn has_quality(&self) -> bool {
        self.inner.lock().unwrap().with_qual()
    }

    #[getter]
    fn records_read(&self) -> u64 {
        self.inner.lock().unwrap().records_read
    }

    #[getter]
    fn bytes_read(&self) -> u64 {
        self.inner.lock().unwrap().bytes_read
    }

    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self.inner.lock().unwrap().schema();
        s.as_ref().to_pyarrow(py)
    }
}

/// Parse "8GB" / "512MB" / "1.5GiB" into bytes.
#[pyfunction]
#[pyo3(name = "parse_size")]
fn parse_size_py(s: &str) -> PyResult<u64> {
    Ok(parse_size(s)?)
}

/// Format a byte count for humans.
#[pyfunction]
#[pyo3(name = "human")]
fn human_py(n: u64) -> String {
    human(n)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<BatchIterator>()?;
    m.add_class::<Budget>()?;
    m.add_class::<PyLease>()?;
    m.add_class::<FastxStream>()?;
    m.add_function(wrap_pyfunction!(parse_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(human_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DATASETS_SERVER", hub::DATASETS_SERVER)?;
    Ok(())
}

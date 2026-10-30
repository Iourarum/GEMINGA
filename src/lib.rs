//! GEMINGA — Memory-Efficient Arrow-Native Dataset Explorer & Reader.
//!
//! Python-facing layer. Every method that touches the network releases the GIL while the
//! tokio runtime does the I/O, and every batch crosses into Python through the Arrow C Data
//! Interface (no copy), so `pyarrow`, `polars`, `pandas`, and `torch` can consume it directly.

mod budget;
mod error;
mod fastx;
mod gem;
mod hub;
mod index;
mod reader;
mod search;
mod trigram;

use arrow_pyarrow::ToPyArrow;
use budget::{human, parse_size, BudgetInner, Lease, Tier};
use error::{GemingaError, Result};
use fastx::{batch_bytes, FastxChunker};
use gem::{gem_extent, read_header, GemChunker, GemIndex};
use index::{SeqBucket, SeqIndex};
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::wrap_pyfunction;
use reader::{
    bytes_per_row, column_stats, describe_file, load_all_metadata, rows_for_bytes, spread_indices,
    ColumnStats, FileInfo, MinMax, MultiFileIter, Source, StreamOptions,
};
use search::{prune, revcomp, SeqHit, SeqSearcher};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::runtime::Runtime;
use trigram::TrigramIndex;

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
        Self {
            rt,
            sources,
            origin,
            partial,
            meta_cache: Mutex::new(None),
            budget: Mutex::new(None),
        }
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

    fn iter_with(
        &self,
        py: Python<'_>,
        opts: StreamOptions,
        only_files: Option<Vec<usize>>,
    ) -> Result<BatchIterator> {
        let metas = self.metas(py)?;
        let (sources, metas): (Vec<Source>, Vec<Option<ArrowReaderMetadata>>) = match only_files {
            Some(idx) => {
                if let Some(bad) = idx.iter().find(|&&i| i >= self.sources.len()) {
                    return Err(GemingaError::Invalid(format!(
                        "file index {bad} out of range"
                    )));
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
            let (d, c, s, t) = (
                dataset.clone(),
                config.clone(),
                split.clone(),
                token.clone(),
            );
            py.detach(move || {
                rt.block_on(hub::list_parquet(
                    &d,
                    c.as_deref(),
                    s.as_deref(),
                    t.as_deref(),
                ))
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
    fn from_urls(
        urls: Vec<String>,
        token: Option<String>,
        sizes: Option<Vec<u64>>,
    ) -> PyResult<Self> {
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
        Ok(Self::new(
            rt,
            sources,
            format!("{} url(s)", urls.len()),
            false,
        ))
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
        Ok(Self::new(
            rt,
            sources,
            format!("{} local file(s)", paths.len()),
            false,
        ))
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
        d.set_item(
            "range_fetches",
            t.range_fetches.load(std::sync::atomic::Ordering::Relaxed),
        )?;
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
        out.set_item(
            "total_row_groups",
            infos.iter().map(|f| f.num_row_groups).sum::<usize>(),
        )?;
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
    #[allow(clippy::too_many_arguments)]
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
                .map(|b| b.default_chunk()),
        };
        let opts = StreamOptions {
            columns,
            row_groups,
            batch_size,
            limit,
            offset,
            chunk_bytes: cb,
        };
        Ok(self.iter_with(py, opts, files)?)
    }

    /// Which row groups could contain `column` values in [lo, hi], from footers alone.
    /// Returns the pruning plan and what it would cost. Nothing is fetched.
    #[pyo3(signature = (column, lo, hi, columns=None))]
    fn prune<'py>(
        &self,
        py: Python<'py>,
        column: &str,
        lo: f64,
        hi: f64,
        columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metas = self.metas(py)?;
        let mut keep_per_file: Vec<Vec<usize>> = Vec::new();
        let (mut cand, mut tot, mut fetch, mut btot) = (0i64, 0i64, 0i64, 0i64);
        let (mut blind, mut rgs, mut trgs) = (0usize, 0usize, 0usize);
        for m in &metas {
            let p = prune(m.metadata(), column, lo, hi, columns.as_ref())?;
            cand += p.candidate_rows;
            tot += p.total_rows;
            fetch += p.bytes_to_fetch;
            btot += p.bytes_total;
            blind += p.kept_without_stats;
            rgs += p.row_groups.len();
            trgs += p.total_row_groups;
            keep_per_file.push(p.row_groups);
        }
        let d = PyDict::new(py);
        let per_file = PyList::empty(py);
        for k in &keep_per_file {
            per_file.append(k.clone())?;
        }
        d.set_item("row_groups_per_file", per_file)?;
        d.set_item("row_groups_kept", rgs)?;
        d.set_item("row_groups_total", trgs)?;
        d.set_item("candidate_rows", cand)?;
        d.set_item("total_rows", tot)?;
        d.set_item("bytes_to_fetch", fetch)?;
        d.set_item("bytes_to_fetch_human", human(fetch.max(0) as u64))?;
        d.set_item("bytes_total", btot)?;
        d.set_item("bytes_total_human", human(btot.max(0) as u64))?;
        d.set_item(
            "selectivity",
            if btot > 0 {
                fetch as f64 / btot as f64
            } else {
                1.0
            },
        )?;
        d.set_item("kept_without_stats", blind)?;
        Ok(d)
    }

    /// Stream only rows whose `column` falls in [lo, hi], fetching only the row groups that
    /// could contain them, then applying the predicate exactly. Returns a list of pyarrow
    /// batches: reading is governed batch by batch, the list itself is the caller's and is
    /// not leased, so a range that keeps most of a large file is better streamed with
    /// `prune()` + `stream(row_groups=...)`.
    #[pyo3(signature = (column, lo, hi, columns=None, batch_size=DEFAULT_BATCH, chunk_bytes=None))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        column: String,
        lo: f64,
        hi: f64,
        columns: Option<Vec<String>>,
        batch_size: usize,
        chunk_bytes: Option<&str>,
    ) -> PyResult<Bound<'py, PyList>> {
        let metas = self.metas(py)?;
        let cb = match chunk_bytes {
            Some(s) => Some(parse_size(s)?),
            None => self
                .budget
                .lock()
                .unwrap()
                .as_ref()
                .map(|b| b.default_chunk()),
        };
        let proj = columns.map(|mut c| {
            if !c.contains(&column) {
                c.push(column.clone());
            }
            c
        });
        let out = PyList::empty(py);
        for (fi, m) in metas.iter().enumerate() {
            let p = prune(m.metadata(), &column, lo, hi, proj.as_ref())?;
            if p.row_groups.is_empty() {
                continue;
            }
            let opts = StreamOptions {
                columns: proj.clone(),
                row_groups: Some(p.row_groups),
                batch_size,
                limit: None,
                offset: None,
                chunk_bytes: cb,
            };
            let b = self.budget.lock().unwrap().clone();
            let mut it =
                MultiFileIter::new(vec![self.sources[fi].clone()], vec![Some(m.clone())], opts)
                    .with_budget(b);
            let rt = self.rt.clone();
            loop {
                let next = py.detach(|| rt.block_on(it.next_batch()))?;
                match next {
                    Some(bt) => {
                        // Row-group statistics only say a group *may* hold matches; the
                        // predicate still has to be applied to the rows themselves.
                        let kept = search::filter_range(&bt, &column, lo, hi)?;
                        if kept.num_rows() > 0 {
                            out.append(kept.to_pyarrow(py)?)?;
                        }
                    }
                    None => break,
                }
            }
        }
        Ok(out)
    }

    /// Attach a Budget. Every batch this dataset produces is then accounted against it,
    /// and batch sizes are derived from it unless you pass an explicit chunk_bytes.
    fn set_budget(&self, budget: Option<PyRef<'_, Budget>>) {
        *self.budget.lock().unwrap() = budget.map(|b| b.inner.clone());
    }

    /// What streaming this dataset would cost, before reading any data: chunk count and the
    /// resident bytes per chunk, for a given budget or explicit chunk size.
    #[pyo3(signature = (chunk_bytes=None, columns=None))]
    fn plan<'py>(
        &self,
        py: Python<'py>,
        chunk_bytes: Option<&str>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let metas = self.metas(py)?;
        let cb = match chunk_bytes {
            Some(s) => parse_size(s)?,
            None => self
                .budget
                .lock()
                .unwrap()
                .as_ref()
                .map(|b| b.default_chunk())
                .unwrap_or(64 << 20),
        };
        let total_rows: i64 = metas
            .iter()
            .map(|m| m.metadata().file_metadata().num_rows())
            .sum();
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
        d.set_item(
            "uncompressed_total_human",
            human((total_rows as f64 * per_row) as u64),
        )?;
        Ok(d)
    }

    /// A representative sample: `n_row_groups` row groups spread evenly across all files.
    /// Reading is governed batch by batch, but the returned list is the caller's and is not
    /// leased — keep `n_row_groups` small, or stream instead.
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
            let b = self.budget.lock().unwrap().clone();
            let mut it = MultiFileIter::new(
                vec![self.sources[fi].clone()],
                vec![Some(metas[fi].clone())],
                opts,
            )
            .with_budget(b);
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
    #[allow(clippy::too_many_arguments)]
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
        let cb = self
            .budget
            .lock()
            .unwrap()
            .as_ref()
            .map(|b| b.default_chunk());
        let opts = StreamOptions {
            columns,
            row_groups,
            batch_size,
            limit,
            offset: None,
            chunk_bytes: cb,
        };
        let it = self.iter_with(py, opts, files)?;
        let rt = self.rt.clone();
        let mut guard = it
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("iterator poisoned"))?;
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
        let mut guard = slf
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("iterator poisoned"))?;
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
                // `inf` means "wait as long as it takes"; a ten-year cap keeps Condvar arithmetic finite
                timeout: Duration::try_from_secs_f64(timeout_s.max(0.0))
                    .unwrap_or(Duration::from_secs(10 * 365 * 24 * 3600))
                    .min(Duration::from_secs(10 * 365 * 24 * 3600)),
            }),
        })
    }

    /// Reserve bytes in a tier. Returns a lease usable as a context manager.
    /// Blocks if the tier is full — with the GIL released, so another Python thread can hand
    /// bytes back — and raises MemoryError if the request can never fit.
    fn acquire(&self, py: Python<'_>, tier: &str, size: &str) -> PyResult<PyLease> {
        let n = parse_size(size)?;
        let inner = self.inner.clone();
        let tier_s = tier.to_string();
        py.detach(move || inner.acquire(&tier_s, n))?;
        Ok(PyLease {
            lease: Mutex::new(Lease::new(Some(self.inner.clone()), tier, n)),
        })
    }

    /// Reserve bytes only if they are free right now: the lease, or None. Never waits, and
    /// the check and the grant are one locked step, so two callers cannot both win.
    fn try_acquire(&self, tier: &str, size: &str) -> PyResult<Option<PyLease>> {
        let n = parse_size(size)?;
        if self.inner.try_acquire(tier, n)? {
            Ok(Some(PyLease {
                lease: Mutex::new(Lease::new(Some(self.inner.clone()), tier, n)),
            }))
        } else {
            Ok(None)
        }
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
            human(r.used),
            human(r.capacity),
            human(v.used),
            human(v.capacity),
            human(r.peak)
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
    fn new(
        path: &str,
        chunk_bytes: &str,
        max_records: usize,
        budget: Option<PyRef<'_, Budget>>,
    ) -> PyResult<Self> {
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
        // Lease the planned chunk *before* reading it (waiting without the GIL), then resize
        // the lease to what was actually materialized.
        let mut lease = Lease::none();
        if let Some(bud) = &slf.budget {
            let planned = slf
                .inner
                .lock()
                .map(|g| g.target_bytes() as u64)
                .unwrap_or(0);
            let b = bud.clone();
            py.detach(move || b.acquire("ram", planned))?;
            lease = Lease::new(Some(bud.clone()), "ram", planned);
        }
        let batch = {
            let mut g = slf
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("reader poisoned"))?;
            let state: &mut FastxChunker = &mut g;
            py.detach(|| state.next_chunk())?
        };
        match batch {
            Some(b) => {
                if slf.budget.is_some() {
                    let sz = batch_bytes(&b);
                    py.detach(|| lease.adjust(sz))?;
                    *slf.lease.lock().unwrap() = lease;
                }
                Ok(Some(b.to_pyarrow(py)?))
            }
            None => {
                lease.release();
                Ok(None)
            }
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

// ---------------------------------------------------------------------------
// Level-of-detail index for sequence files
// ---------------------------------------------------------------------------

fn bucket_dict<'py>(py: Python<'py>, b: &SeqBucket) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("start", b.start_record)?;
    d.set_item("end", b.end_record)?;
    d.set_item("n", b.n)?;
    d.set_item("bases", b.bases)?;
    d.set_item("min_len", b.min_len)?;
    d.set_item("max_len", b.max_len)?;
    d.set_item("mean_len", b.mean_len())?;
    d.set_item("gc", b.gc)?;
    d.set_item("n_frac", b.n_frac)?;
    d.set_item(
        "mean_q",
        if b.mean_q.is_finite() {
            Some(b.mean_q)
        } else {
            None
        },
    )?;
    d.set_item("byte_offset", b.byte_offset)?;
    Ok(d)
}

/// A small sidecar that makes a sequence file zoomable and seekable.
/// One pass to build; every later navigation is arithmetic on the index.
#[pyclass(module = "geminga._core", name = "SeqIndex")]
pub struct PySeqIndex {
    inner: SeqIndex,
}

#[pymethods]
impl PySeqIndex {
    /// Build by scanning the file once. Releases the GIL for the whole scan.
    #[staticmethod]
    #[pyo3(signature = (path, n_buckets=2048))]
    fn build(py: Python<'_>, path: String, n_buckets: usize) -> PyResult<Self> {
        let inner = py.detach(move || SeqIndex::build(&path, n_buckets))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: SeqIndex::load(&path)?,
        })
    }

    fn save(&self, path: String) -> PyResult<()> {
        Ok(self.inner.save(&path)?)
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }
    #[getter]
    fn n_records(&self) -> u64 {
        self.inner.n_records
    }
    #[getter]
    fn total_bases(&self) -> u64 {
        self.inner.total_bases
    }
    #[getter]
    fn file_bytes(&self) -> u64 {
        self.inner.file_bytes
    }
    #[getter]
    fn has_quality(&self) -> bool {
        self.inner.has_quality
    }
    #[getter]
    fn seekable(&self) -> bool {
        self.inner.seekable
    }
    #[getter]
    fn bytes_per_record(&self) -> f64 {
        self.inner.bytes_per_record
    }
    #[getter]
    fn n_buckets(&self) -> usize {
        self.inner.buckets.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SeqIndex({}, {} records, {} buckets, {} on disk)",
            self.inner.path,
            self.inner.n_records,
            self.inner.buckets.len(),
            human(self.inner.file_bytes)
        )
    }

    /// Re-aggregate the index over [lo, hi) into `target` buckets. This is the zoom operation:
    /// no file access, pure arithmetic, so it is instant at any zoom level.
    #[pyo3(signature = (lo=0, hi=None, target=256))]
    fn zoom<'py>(
        &self,
        py: Python<'py>,
        lo: u64,
        hi: Option<u64>,
        target: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let hi = hi.unwrap_or(self.inner.n_records);
        let out = PyList::empty(py);
        for b in self.inner.zoom(lo, hi, target) {
            out.append(bucket_dict(py, &b)?)?;
        }
        Ok(out)
    }

    /// Resident and transfer cost of materializing records [lo, hi), before doing it.
    #[pyo3(signature = (lo=0, hi=None))]
    fn cost<'py>(&self, py: Python<'py>, lo: u64, hi: Option<u64>) -> PyResult<Bound<'py, PyDict>> {
        let hi = hi.unwrap_or(self.inner.n_records).min(self.inner.n_records);
        let n = hi.saturating_sub(lo);
        let payload = (n as f64 * self.inner.bytes_per_record) as u64;
        // Arrow string arrays add 32-bit offsets per column plus the id/seq/qual buffers.
        let arrow = payload + n * if self.inner.has_quality { 20 } else { 12 };
        let d = PyDict::new(py);
        d.set_item("records", n)?;
        d.set_item("transfer_bytes", payload)?;
        d.set_item("resident_bytes", arrow)?;
        d.set_item("resident_human", human(arrow))?;
        d.set_item(
            "fraction_of_file",
            if self.inner.file_bytes > 0 {
                payload as f64 / self.inner.file_bytes as f64
            } else {
                0.0
            },
        )?;
        let (ck_rec, ck_off) = self.inner.checkpoint_for(lo);
        d.set_item("seek_from_record", ck_rec)?;
        d.set_item("seek_byte", ck_off)?;
        d.set_item("skip_records", lo.saturating_sub(ck_rec))?;
        d.set_item("seekable", self.inner.seekable)?;
        Ok(d)
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

fn hit_dict<'py>(py: Python<'py>, h: &SeqHit) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("record", h.record)?;
    d.set_item("pattern", h.pattern)?;
    d.set_item("offset", h.offset)?;
    d.set_item("id", &h.id)?;
    Ok(d)
}

/// Multi-pattern sequence search over a FASTA/FASTQ, streamed under a budget.
/// All patterns are matched in a single pass, so 300 adapters cost about what one costs.
#[pyfunction]
#[pyo3(signature = (path, patterns, both_strands=true, max_hits=100_000, lo=0, hi=None, first_per_record=false))]
#[allow(clippy::too_many_arguments)]
fn search_fastx<'py>(
    py: Python<'py>,
    path: String,
    patterns: Vec<String>,
    both_strands: bool,
    max_hits: usize,
    lo: u64,
    hi: Option<u64>,
    first_per_record: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let searcher = SeqSearcher::new(&patterns, both_strands)?;
    let hi = hi.unwrap_or(u64::MAX);
    let (hits, scanned, matched_records, bases) = py.detach(move || -> Result<_> {
        let mut reader = needletail::parse_fastx_file(&path)
            .map_err(|e| GemingaError::Invalid(format!("cannot open {path}: {e}")))?;
        let mut out: Vec<SeqHit> = Vec::new();
        let mut idx = 0u64;
        let mut scanned = 0u64;
        let mut matched = 0u64;
        let mut bases = 0u64;
        while let Some(rec) = reader.next() {
            let rec = rec.map_err(|e| GemingaError::Invalid(format!("{e}")))?;
            if idx >= hi || out.len() >= max_hits {
                break;
            }
            if idx >= lo {
                let seq = rec.raw_seq();
                let id = String::from_utf8_lossy(rec.id()).to_string();
                let before = out.len();
                searcher.find(seq, idx, &id, &mut out, first_per_record);
                if out.len() > before {
                    matched += 1;
                }
                scanned += 1;
                bases += seq.len() as u64;
            }
            idx += 1;
        }
        Ok((out, scanned, matched, bases))
    })?;

    let list = PyList::empty(py);
    for h in &hits {
        list.append(hit_dict(py, h)?)?;
    }
    let d = PyDict::new(py);
    d.set_item("hits", list)?;
    d.set_item("n_hits", hits.len())?;
    d.set_item("records_scanned", scanned)?;
    d.set_item("records_matched", matched_records)?;
    d.set_item("bases_scanned", bases)?;
    d.set_item("truncated", hits.len() >= max_hits)?;
    Ok(d)
}

/// Reverse complement of a DNA/RNA string.
#[pyfunction]
#[pyo3(name = "revcomp")]
fn revcomp_py(s: &str) -> String {
    revcomp(s)
}

// ---------------------------------------------------------------------------
// Stereo-seq GEM (text expression matrix)
// ---------------------------------------------------------------------------

/// Byte-chunked GEM reader yielding pyarrow.RecordBatch (gene, x, y, count[, exon]).
/// gzip handled transparently; an optional spatial window is applied during parsing, so
/// filtered-out rows never enter an Arrow buffer.
#[pyclass(module = "geminga._core")]
pub struct GemStream {
    inner: Mutex<GemChunker>,
    budget: Option<Arc<BudgetInner>>,
    lease: Mutex<Lease>,
    window: Option<(i32, i32, i32, i32)>,
}

#[pymethods]
impl GemStream {
    #[new]
    #[pyo3(signature = (path, chunk_bytes="64MB", window=None, budget=None, apply_offset=true))]
    fn new(
        path: &str,
        chunk_bytes: &str,
        window: Option<(i32, i32, i32, i32)>,
        budget: Option<PyRef<'_, Budget>>,
        apply_offset: bool,
    ) -> PyResult<Self> {
        let target = parse_size(chunk_bytes)? as usize;
        Ok(Self {
            inner: Mutex::new(GemChunker::new(path, target, apply_offset)?),
            budget: budget.map(|b| b.inner.clone()),
            lease: Mutex::new(Lease::none()),
            window,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(slf: PyRef<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = slf.py();
        slf.lease.lock().unwrap().release();
        let win = slf.window;
        loop {
            let mut lease = Lease::none();
            if let Some(bud) = &slf.budget {
                let planned = slf
                    .inner
                    .lock()
                    .map(|g| g.target_bytes() as u64)
                    .unwrap_or(0);
                let b = bud.clone();
                py.detach(move || b.acquire("ram", planned))?;
                lease = Lease::new(Some(bud.clone()), "ram", planned);
            }
            let batch = {
                let mut g = slf
                    .inner
                    .lock()
                    .map_err(|_| PyRuntimeError::new_err("reader poisoned"))?;
                let state: &mut GemChunker = &mut g;
                py.detach(|| state.next_chunk(win))?
            };
            match batch {
                Some(b) if b.num_rows() == 0 => {
                    lease.release();
                    continue; // whole chunk filtered out; keep going
                }
                Some(b) => {
                    if slf.budget.is_some() {
                        let sz = batch_bytes(&b);
                        py.detach(|| lease.adjust(sz))?;
                        *slf.lease.lock().unwrap() = lease;
                    }
                    return Ok(Some(b.to_pyarrow(py)?));
                }
                None => {
                    lease.release();
                    return Ok(None);
                }
            }
        }
    }

    #[getter]
    fn rows_read(&self) -> u64 {
        self.inner.lock().unwrap().rows_read
    }
    #[getter]
    fn bytes_read(&self) -> u64 {
        self.inner.lock().unwrap().bytes_read
    }

    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let g = self.inner.lock().unwrap();
        let d = PyDict::new(py);
        for (k, v) in &g.header.fields {
            d.set_item(k, v)?;
        }
        d.set_item("columns", g.header.columns.clone())?;
        d.set_item("bin_size", g.header.bin_size())?;
        d.set_item("offset_x", g.header.offset_x())?;
        d.set_item("offset_y", g.header.offset_y())?;
        d.set_item("chip", g.header.chip())?;
        Ok(d)
    }
}

/// Read only a GEM's `#` header — a few hundred bytes, whatever the file size.
#[pyfunction]
#[pyo3(name = "gem_header")]
fn gem_header_py<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let h = read_header(path)?;
    let d = PyDict::new(py);
    for (k, v) in &h.fields {
        d.set_item(k, v)?;
    }
    d.set_item("columns", h.columns.clone())?;
    d.set_item("bin_size", h.bin_size())?;
    d.set_item("offset_x", h.offset_x())?;
    d.set_item("offset_y", h.offset_y())?;
    d.set_item("chip", h.chip())?;
    d.set_item("data_offset", h.data_offset)?;
    Ok(d)
}

/// One scan of a GEM to learn its extent, row count and gene count. GIL released throughout.
#[pyfunction]
#[pyo3(name = "gem_extent")]
fn gem_extent_py<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyDict>> {
    let (x0, x1, y0, y1, rows, genes) = py.detach(move || gem_extent(&path))?;
    let d = PyDict::new(py);
    d.set_item("min_x", x0)?;
    d.set_item("max_x", x1)?;
    d.set_item("min_y", y0)?;
    d.set_item("max_y", y1)?;
    d.set_item("width", x1 - x0)?;
    d.set_item("height", y1 - y0)?;
    d.set_item("rows", rows)?;
    d.set_item("genes", genes)?;
    Ok(d)
}

/// One pass over a GEM produces everything needed to navigate it afterwards: chip extent,
/// a spatial density pyramid, per-gene totals, and the cost model's row estimates.
#[pyclass(module = "geminga._core", name = "GemIndex")]
pub struct PyGemIndex {
    inner: GemIndex,
}

#[pymethods]
impl PyGemIndex {
    #[staticmethod]
    #[pyo3(signature = (path, grid=512))]
    fn build(py: Python<'_>, path: String, grid: usize) -> PyResult<Self> {
        let inner = py.detach(move || GemIndex::build(&path, grid))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: GemIndex::load(&path)?,
        })
    }
    fn save(&self, path: String) -> PyResult<()> {
        Ok(self.inner.save(&path)?)
    }

    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }
    #[getter]
    fn n_rows(&self) -> u64 {
        self.inner.n_rows
    }
    #[getter]
    fn total_mid(&self) -> u64 {
        self.inner.total_mid
    }
    #[getter]
    fn bin_size(&self) -> u32 {
        self.inner.bin_size
    }
    #[getter]
    fn file_bytes(&self) -> u64 {
        self.inner.file_bytes
    }
    #[getter]
    fn grid(&self) -> usize {
        self.inner.grid
    }
    #[getter]
    fn has_exon(&self) -> bool {
        self.inner.has_exon
    }
    #[getter]
    fn chip(&self) -> Option<String> {
        self.inner.chip.clone()
    }
    #[getter]
    fn n_genes(&self) -> usize {
        self.inner.genes.len()
    }
    #[getter]
    fn genes(&self) -> Vec<String> {
        self.inner.genes.clone()
    }
    #[getter]
    fn bytes_per_row(&self) -> f64 {
        self.inner.bytes_per_row
    }

    /// (min_x, max_x, min_y, max_y) in chip coordinates.
    #[getter]
    fn extent(&self) -> (i32, i32, i32, i32) {
        (
            self.inner.min_x,
            self.inner.max_x,
            self.inner.min_y,
            self.inner.max_y,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "GemIndex({}, bin{}, {} rows, {} genes, x {}-{} y {}-{}, {} on disk)",
            self.inner.path,
            self.inner.bin_size,
            self.inner.n_rows,
            self.inner.genes.len(),
            self.inner.min_x,
            self.inner.max_x,
            self.inner.min_y,
            self.inner.max_y,
            human(self.inner.file_bytes)
        )
    }

    /// Observations inside a spatial window, from the occupancy grid. Costs nothing.
    fn rows_in(&self, x0: i32, x1: i32, y0: i32, y1: i32) -> u64 {
        self.inner.rows_in(x0, x1, y0, y1)
    }

    /// The density grid re-binned over a window: the zoom operation, pure arithmetic.
    /// Returns (side, flat row-major values).
    #[pyo3(signature = (x0, x1, y0, y1, target=256))]
    fn zoom(&self, x0: i32, x1: i32, y0: i32, y1: i32, target: usize) -> (usize, Vec<f32>) {
        self.inner.zoom(x0, x1, y0, y1, target)
    }

    /// Cost of materializing a spatial window, before doing it.
    fn cost<'py>(
        &self,
        py: Python<'py>,
        x0: i32,
        x1: i32,
        y0: i32,
        y1: i32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rows = self.inner.rows_in(x0, x1, y0, y1);
        // Arrow: dictionary key + two u32 coords + count, plus the shared gene dictionary.
        let resident = rows * 16;
        let transfer = (rows as f64 * self.inner.bytes_per_row) as u64;
        let d = PyDict::new(py);
        d.set_item("rows", rows)?;
        d.set_item("resident_bytes", resident)?;
        d.set_item("resident_human", human(resident))?;
        d.set_item("transfer_bytes", transfer)?;
        d.set_item(
            "fraction_of_rows",
            rows as f64 / self.inner.n_rows.max(1) as f64,
        )?;
        Ok(d)
    }

    #[pyo3(signature = (k=20))]
    fn top_genes(&self, k: usize) -> Vec<(String, u64)> {
        self.inner.top_genes(k)
    }
}

/// A trigram index over a line-oriented text file (a journal, a catalog, any JSONL log): the
/// shape of tgrep. Built once, extended as the file grows, searched by literal in the blocks
/// that can hold it. `search` also scans the lines appended since the last `update`.
#[pyclass(module = "geminga._core", name = "TrigramIndex")]
pub struct PyTrigramIndex {
    inner: TrigramIndex,
}

#[pymethods]
impl PyTrigramIndex {
    /// Index `path` in blocks of `block_lines` lines. Releases the GIL for the scan.
    #[staticmethod]
    #[pyo3(signature = (path, block_lines=64))]
    fn build(py: Python<'_>, path: String, block_lines: u32) -> PyResult<Self> {
        let inner = py.detach(move || TrigramIndex::build(&path, block_lines))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn load(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: TrigramIndex::load(&path)?,
        })
    }

    fn save(&self, path: String) -> PyResult<()> {
        Ok(self.inner.save(&path)?)
    }

    /// Index the lines appended since; returns how many. Releases the GIL.
    fn update(&mut self, py: Python<'_>) -> PyResult<u64> {
        let mut inner = self.inner.clone();
        let (n, inner) = py.detach(move || -> Result<(u64, TrigramIndex)> {
            let n = inner.update()?;
            Ok((n, inner))
        })?;
        self.inner = inner;
        Ok(n)
    }

    /// `[(line_number, text), ...]` for lines containing `pattern`; line numbers from 0.
    #[pyo3(signature = (pattern, ignore_case=true))]
    fn search(
        &self,
        py: Python<'_>,
        pattern: String,
        ignore_case: bool,
    ) -> PyResult<Vec<(u64, String)>> {
        let inner = self.inner.clone();
        Ok(py.detach(move || inner.search(&pattern, ignore_case))?)
    }

    /// Block ids that can hold `pattern`, or None when it is shorter than a trigram.
    fn candidates(&self, pattern: String) -> Option<Vec<u32>> {
        let pat: Vec<u8> = pattern.bytes().map(|b| b.to_ascii_lowercase()).collect();
        self.inner.candidates(&pat)
    }

    #[getter]
    fn source_path(&self) -> String {
        self.inner.source.clone()
    }
    #[getter]
    fn indexed_bytes(&self) -> u64 {
        self.inner.indexed_bytes
    }
    #[getter]
    fn n_lines(&self) -> u64 {
        self.inner.n_lines
    }
    #[getter]
    fn n_blocks(&self) -> usize {
        self.inner.n_blocks()
    }
    #[getter]
    fn block_lines(&self) -> u32 {
        self.inner.block_lines
    }
    #[getter]
    fn n_trigrams(&self) -> usize {
        self.inner.n_trigrams()
    }
    /// Trigrams present in every block, kept as a mark rather than a list (a JSON log's keys).
    #[getter]
    fn n_everywhere(&self) -> usize {
        self.inner.n_everywhere()
    }
    #[getter]
    fn postings_bytes(&self) -> usize {
        self.inner.postings_bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrigramIndex({}, {} lines in {} blocks, {} trigrams, {} of postings)",
            self.inner.source,
            self.inner.n_lines,
            self.inner.n_blocks(),
            self.inner.n_trigrams(),
            human(self.inner.postings_bytes() as u64)
        )
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Dataset>()?;
    m.add_class::<BatchIterator>()?;
    m.add_class::<Budget>()?;
    m.add_class::<PyLease>()?;
    m.add_class::<FastxStream>()?;
    m.add_class::<PySeqIndex>()?;
    m.add_class::<PyGemIndex>()?;
    m.add_class::<GemStream>()?;
    m.add_class::<PyTrigramIndex>()?;
    m.add_function(wrap_pyfunction!(gem_header_py, m)?)?;
    m.add_function(wrap_pyfunction!(gem_extent_py, m)?)?;
    m.add_function(wrap_pyfunction!(search_fastx, m)?)?;
    m.add_function(wrap_pyfunction!(revcomp_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(human_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("DATASETS_SERVER", hub::DATASETS_SERVER)?;
    Ok(())
}

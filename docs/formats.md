# Formats

Everything tabular arrives as a zero-copy `pyarrow.RecordBatch`; imaging arrives as `numpy`.

```{list-table}
:header-rows: 1

* - Format
  - Backend
  - Chunk unit
  - Notes
* - Stereo-seq GEF
  - Python — `h5py`
  - bin level + spatial window
  - `bin1`…`bin200` are the zoom levels; gene index gives contiguous slices
* - Stereo-seq GEM
  - Rust
  - byte-sized row runs
  - dictionary-encoded genes; spatial index sidecar
* - Parquet (local, HTTP, HF Hub)
  - Rust — `arrow`/`parquet` + `object_store`
  - row groups
  - HTTP range requests; footer-only inspection reads ~0.3 % of a file
* - FASTA / FASTQ (`.gz` ok)
  - Rust — `needletail`
  - byte-sized record runs
  - ~680 MB/s plain, ~90 MB/s gzipped, single core
* - h5ad (scRNA-seq, scATAC-seq)
  - Python — `h5py`
  - CSR row blocks (cells)
  - block size from `indptr`; CSC rejected with a fix
* - Arrow IPC / Feather
  - Python — `pyarrow`
  - record batches
  - memory-mapped
* - TIFF / OME-TIFF
  - Python — `tifffile` + native tile reader
  - image tiles, optional overlap
  - decodes only the tiles a window touches; no zarr dependency
* - Zarr / OME-Zarr
  - Python — `zarr`
  - image tiles
  -
```

## Adding a format

The contract is three methods. `H5adReader` in `python/geminga/formats.py` is the shortest
complete example.

1. **Plan before reading.** Produce a `Plan` (or a `cost()` dict) from metadata alone.
2. **Acquire before materializing.** Take from the budget, sized to what you are about to
   decode.
3. **Release on advance.** Free when the consumer asks for the next chunk.

```python
class MyReader(_Governed):
    def plan(self) -> Plan: ...
    def __iter__(self):
        for window in self.windows():
            self._swap_lease(self.bytes_for(window))
            yield self.read(window)
```

Every new reader gets a test asserting peak budget use stays under the declared cap. That test
is the point of the library.

## Sinks

Results go out as they are produced, so nothing accumulates.

```python
with g.ParquetSink("out.parquet") as sink:
    for batch in reader:
        sink.write(transform(batch))
```

`MaskSink` writes per-tile segmentation straight to a full-resolution Zarr array, offsetting
instance labels so ids stay unique across the image:

```python
masks = g.MaskSink("masks.zarr", shape=(img.shape[0], img.shape[1]))
for tile, window in reader:
    masks.write(model(tile), window)
print(masks.n_objects)
```

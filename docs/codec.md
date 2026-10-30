# Compressed resident blocks

A budget says how many bytes may be resident. Compression changes what a byte buys. The
integers an omics matrix is made of — UMI counts, sorted column indices, row lengths, bin
coordinates — pack four to ten times smaller than their int32 containers, and if the packed
form is what sits in RAM or VRAM and decoding happens where the data lives, a 4 GB laptop GPU
or a 4 GB slice of a shared server GPU holds four to ten times more of the dataset. GEMINGA's
job is the bookkeeping: count the packed bytes against the budget, charge the decoded bytes
only while someone holds them, and keep the codec pluggable so a better one — the GPU
compressor being written for the shared server — drops in without touching readers or views.

## The contract

```python
class Codec(Protocol):
    name: str
    def encode(self, arr: np.ndarray) -> Encoded: ...
    def decode(self, enc: Encoded) -> np.ndarray: ...
    # optional: decode straight into a wgpu buffer, so the raw bytes never cross the bus
    def decode_device(self, enc: Encoded, device, out_buffer=None): ...
    def upload(self, enc: Encoded, device) -> Encoded: ...
```

An `Encoded` carries the payload (host arrays, bytes, or device buffers once uploaded), its
packed size, the raw size, dtype and shape, and whatever `meta` decoding needs. Everything
else — the store, the budget, the readers — only ever asks `nbytes`, `raw_bytes` and `ratio`.

## The reference codec: per-block bit packing

`BitpackCodec` is the one worth reading. It takes any integer array whose values fit 32 bits,
cuts it into blocks of 128, and per block chooses the cheaper of two codings:

- **frame of reference** — `value − block minimum`, packed at the block's bit width;
- **zigzag delta** — differences from the previous value with the sign folded into the low
  bit, packed at the width of the largest difference.

Both are decodable by one GPU thread per block: the WGSL kernel in `WGSL_BITPACK_DECODE`
unpacks lanes from the word stream and either adds the base or prefix-sums from it. The host
decoder is vectorised numpy over the same bitstream, so host and device are interchangeable
and are tested against each other bit for bit.

What it gives, on one laptop core (`examples/bench_codec.py`, 1 M elements each):

| array | ratio | mean width | encode | decode |
|---|---:|---:|---:|---:|
| UMI counts, Poisson(2)+1 | 7.8× | 3.1 bits | 53 MB/s | 116 MB/s |
| sorted column indices over 30 k genes | 10.7× | 2.0 bits | 71 MB/s | 101 MB/s |
| CSR indices, ~300 nnz per row | 2.4× | 12.3 bits | 57 MB/s | 101 MB/s |
| row lengths | 3.2× | 9.0 bits | 58 MB/s | 112 MB/s |
| Stereo-seq bin x coordinates | 10.7× | 2.0 bits | 69 MB/s | 112 MB/s |
| random int32 | 1.0× | 31 bits | 52 MB/s | 136 MB/s |

The weak row is honest: CSR indices reset at every row boundary, and a 128-block that
straddles one pays a wide width for the whole block. A row-aware scheme (delta within rows,
using the lengths the store already has) would roughly double that ratio — this is the first
thing a plug-in codec should beat. Floats are not packed at all; they go through
`BytesCodec` (zstd, lz4 or zlib, whichever is installed), which a GPU cannot decode in
parallel. Normalised expression is therefore not where the gain is; raw counts are.

The GPU kernel is tested here on a software adapter (llvmpipe), which only proves it correct.
On real hardware the decode is memory-bound: a thread per block, coalesced reads, and a
serial prefix sum of 128 values.

## The store

```python
b = g.Budget(ram="8MB")
store = g.compressed_store(b, tier="ram")
for i, blk in enumerate(g.H5adReader("atlas.h5ad", budget=b)):
    store.put(f"blk{i}", blk["X"])              # leases the packed bytes
with store.get("blk3") as X:                     # scratch lease for the raw bytes while held
    ...
store.print_report()
```

```
compressed store (ram, bitpack): 23 block(s), 23.0 MB represented in 6.2 MB (3.7x);
tier capacity 8.0 MB now holds ~29.7 MB of data
```

That is the sc.h5ad fixture: the whole 23 MB matrix resident, packed, under an 8 MB budget
whose peak stayed at 7.9 MB. The accounting rule is the point: a block costs its packed size
while stored and its raw size only while a caller holds the decoded copy; `report()` says what
the tier now effectively holds. A block that does not fit packed is a `MemoryError` before
anything is stored, like any other lease.

Sparse matrices are stored as three arrays — row lengths (smaller than `indptr`), indices and
data — each packed on its own; integer-valued float data is packed as integers and restored to
its dtype. `encode_csr` / `decode_csr` are the standalone form.

### On the device

```python
store = g.compressed_store(budget, tier="vram", device=wgpu_device)
store.put("counts", counts)                      # packed, uploaded; vram leased for the packed bytes
with store.get_device("counts") as da:           # decoded on the GPU; vram leased for the raw bytes
    pass_.set_bind_group(0, bind(da.buffer))     # use it in place
    da.read()                                    # or copy it back
```

With `tier="vram"` the packed payload lives in device buffers and the host keeps nothing.
`get_device()` runs the kernel into a buffer the caller uses directly — in a compute pass, or
in the Stage — and `get()` still works: it decodes on the device and reads the result back.
A reservation from the [shared-GPU server](reserve.md) gives a budget whose `vram` capacity is
the slice granted, so the store cannot exceed it.

## Plugging in your own compressor

The contract is small on purpose, and there is a conformance test so you do not have to guess:

```python
class MyCodec:
    name = "mine"
    def encode(self, arr) -> Encoded: ...        # raw_bytes, nbytes, dtype, shape, payload, meta
    def decode(self, enc) -> np.ndarray: ...     # exact for what encode() accepted
    # optional, for the GPU path
    def upload(self, enc, device) -> Encoded: ...            # payload to device buffers; nbytes unchanged
    def decode_device(self, enc, device, out_buffer=None): ...  # a buffer holding the decoded array

g.check_codec(MyCodec())                  # raises CodecError naming the case that failed
g.check_codec(MyCodec(), device=dev)      # ...and checks the device decoder against the host one
```

`check_codec` runs the arrays the store relies on — UMI counts, sorted indices, negative
values, int32/uint32 extremes, narrow and 64-bit integers, 2-D, empty, floats, strings — and
reports what was handled, what the codec *declined* (a `TypeError`/`ValueError` from
`encode`, which is fine: the store falls back to bytes for those) and what *failed*. A GPU
decoder declares what its output buffer holds in `enc.meta["device_layout"]`: `"lanes-u32"`
(one u32 per element, the bitpack kernel's convention) or `"native"` (the array's own dtype,
byte for byte). `DeviceArray.read()` and the store honour either.

Register it, or ship it as a package with an entry point and refer to it by name:

```python
g.register_codec("mine", MyCodec)                 # this process
store = g.compressed_store(budget, codec="mine")

# pyproject.toml of your package
[project.entry-points."geminga.codecs"]
mine = "mypackage.codec:MyCodec"
```

`examples/plugin_codec.py` is a complete worked example (run-length encoding) with the
ratio table that shows where it beats the built-in packer and where it does not.

## Tables, and DuckDB

A table is packed column by column — integers through the packer, floats and strings through
bytes — and comes back as a pyarrow Table, so a DuckDB connection can query packed blocks:

```python
st.put("hits", arrow_table)                  # or a dict of columns
with st.get_table("hits") as t: ...          # pyarrow.Table, raw bytes leased while held
holds = st.duckdb(con)                       # registers every packed table as a view
con.execute("select gene, sum(count) from hits group by gene")
for h in holds.values(): h.release()        # give the decoded bytes back
```

Nulls are refused rather than silently packed as zeros; fill them first.

## What it is not

Not a general compressor, not a file format, and not transparent to the readers: a block
goes into the store because the analyst (or a pipeline) puts it there, and comes out through
`get()`. Making the views themselves keep their windows packed — zoom out, keep the coarse
level resident packed, zoom back in without re-reading — is the next step, and the reason
the codec contract exists before the feature does.

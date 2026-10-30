"""Compressed resident blocks: what a byte of budget buys when the bytes are packed.

A budget says how many bytes may be resident. The integers an omics matrix is made of — UMI
counts, sorted column indices, row lengths, bin coordinates — pack four to ten times smaller
than their int32 containers. If the packed form is what sits in RAM or VRAM and decoding
happens where the data lives, a 4 GB laptop GPU or a 4 GB slice of a shared server GPU holds
four to ten times more of the dataset. GEMINGA's job here is the bookkeeping, and keeping the
codec pluggable so a better one — the GPU compressor written for the shared server — drops
in without touching readers or views.

Three parts:

* the **contract** (:class:`Codec`, :class:`Encoded`): encode an array, decode it, and —
  optionally — decode it straight into a device buffer so the raw bytes never cross the bus;
* the **reference codec** (:class:`BitpackCodec`): per-block frame-of-reference or zigzag-delta
  bit packing of 32-bit integers, with a host decoder in numpy and the same bitstream decoded
  by one GPU thread per block (``WGSL_BITPACK_DECODE``), tested against each other bit for bit;
  :class:`BytesCodec` (zstd / lz4 / zlib) for everything that is not an integer;
* the **store** (:class:`CompressedStore`): blocks resident packed, charged to a budget tier at
  their packed size, charged at their raw size only while a caller holds the decoded copy.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ._core import Budget, human

__all__ = ["Codec", "Encoded", "BitpackCodec", "BytesCodec", "CompressedStore", "StoreEntry", "DeviceArray",
           "compressed_store", "encode_csr", "decode_csr", "ratio_report", "bitpack", "bytes_codec",
           "BLOCK", "WGSL_BITPACK_DECODE", "register_codec", "get_codec", "available_codecs", "check_codec",
           "CodecError"]

#: Elements per packed block: one GPU thread decodes one block.
BLOCK = 128
_MODE_FOR, _MODE_DELTA = 0, 1


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

@dataclass
class Encoded:
    """One array, packed. `payload` holds host arrays (or device buffers after `upload`)."""
    codec: str
    dtype: str
    shape: tuple
    nbytes: int                       # packed bytes — what the store charges
    raw_bytes: int                    # what the decoded copy costs
    payload: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    on_device: bool = False

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(self.nbytes, 1)

    @property
    def n(self) -> int:
        return int(self.meta.get("n", int(np.prod(self.shape)) if self.shape else 1))

    def __repr__(self) -> str:
        where = "device" if self.on_device else "host"
        return (f"Encoded({self.codec}, {self.dtype}{list(self.shape)}, {human(self.raw_bytes)} → "
                f"{human(self.nbytes)}, x{self.ratio:.1f}, {where})")


@runtime_checkable
class Codec(Protocol):
    """What a codec must provide. Everything else only asks `nbytes`, `raw_bytes`, `ratio`."""
    name: str

    def encode(self, arr: np.ndarray) -> Encoded: ...

    def decode(self, enc: Encoded) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

def _lanes_of(arr: np.ndarray) -> tuple:
    """Integer array → (int64 values, numpy dtype). Values must fit int32 (signed) or uint32."""
    a = np.asarray(arr)
    if a.dtype.kind not in "iu":
        raise TypeError(f"bitpack packs integers; got {a.dtype} (use BytesCodec for floats and strings)")
    v = a.reshape(-1).astype(np.int64, copy=False)
    if a.dtype.kind == "i":
        if v.size and (v.min() < -(2 ** 31) or v.max() > 2 ** 31 - 1):
            raise ValueError("bitpack needs values in int32 range")
    else:
        if v.size and v.max() > 2 ** 32 - 1:
            raise ValueError("bitpack needs values in uint32 range")
    return v, a.dtype


def _lanes_to_dtype(words: np.ndarray, dtype, shape) -> np.ndarray:
    """u32 lanes back into the array's own dtype and shape (two's complement for signed)."""
    dt = np.dtype(dtype)
    w = np.asarray(words, dtype=np.uint32)
    if dt.kind == "u":
        out = w.astype(dt, copy=False) if dt.itemsize >= 4 else w.astype(dt)
    else:
        out = w.view(np.int32).astype(dt, copy=False)
    return np.array(out, copy=True).reshape(shape)


def _widths(x: np.ndarray) -> np.ndarray:
    """Bits needed per block for the non-negative int64 values in x (0 for 0)."""
    x = np.asarray(x, np.int64)
    out = np.zeros(x.shape, np.int64)
    nz = x > 0
    out[nz] = np.floor(np.log2(x[nz].astype(np.float64))).astype(np.int64) + 1
    # log2 can round across a power of two: correct the few that need it
    too_small = nz & (out < 64) & ((np.int64(1) << np.minimum(out, 62)) <= x)
    out[too_small] += 1
    too_big = nz & (out > 1) & ((np.int64(1) << np.minimum(out - 1, 62)) > x)
    out[too_big] -= 1
    return out


def _pack_blocks(vals: np.ndarray, widths: np.ndarray, counts: np.ndarray) -> tuple:
    """Pack the (n_blocks × BLOCK) matrix of non-negative lanes, each block at its own width,
    into one little-endian u32 word stream. Returns (words, word offset per block)."""
    nwords = (counts.astype(np.int64) * widths.astype(np.int64) + 31) // 32
    offsets = np.concatenate([[0], np.cumsum(nwords)[:-1]]).astype(np.int64)
    total = int(nwords.sum())
    if total == 0:
        return np.zeros(0, np.uint32), offsets
    # bit position of every lane: block word offset * 32 + i * width
    lane = np.arange(BLOCK, dtype=np.int64)[None, :]
    pos = offsets[:, None] * 32 + lane * widths.astype(np.int64)[:, None]        # (n_blocks, BLOCK)
    live = (lane < counts[:, None]) & (widths[:, None] > 0)
    v = vals[live].astype(np.uint64)
    pos = pos[live]
    bo = (pos & 31).astype(np.uint64)
    wi = pos >> 5
    words = np.zeros(total + 1, np.uint64)                      # +1: a lane may spill into the next word
    np.add.at(words, wi, (v << bo) & np.uint64(0xFFFFFFFF))
    np.add.at(words, wi + 1, v >> (np.uint64(32) - bo))         # bo == 0 shifts by 32: zero for < 2^32
    return words[:total].astype(np.uint32), offsets


def _unpack_blocks(words: np.ndarray, offsets: np.ndarray, widths: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The inverse of `_pack_blocks`: (n_blocks × BLOCK) int64 lanes (zero beyond each count)."""
    lane = np.arange(BLOCK, dtype=np.int64)[None, :]
    pos = offsets.astype(np.int64)[:, None] * 32 + lane * widths.astype(np.int64)[:, None]
    wi = pos >> 5
    bo = (pos & 31).astype(np.uint64)
    w = np.concatenate([words.astype(np.uint64), np.zeros(2, np.uint64)])
    wi = np.minimum(wi, w.size - 2)
    lo = w[wi] >> bo
    hi = np.where(bo > 0, w[wi + 1] << (np.uint64(32) - bo), np.uint64(0))
    mask = (np.uint64(1) << widths.astype(np.uint64))[:, None] - np.uint64(1)
    out = ((lo | hi) & mask).astype(np.int64)
    out[(lane >= counts[:, None]) | (widths[:, None] == 0)] = 0
    return out


class BitpackCodec:
    """Per-block bit packing of 32-bit integers: frame of reference or zigzag delta, whichever
    is narrower for that block of 128. One GPU thread decodes one block.

    Table per block (u32 ×4): word offset, base lane (block minimum, or the first value in
    delta mode), ``mode << 8 | width``, count. Words: the packed lanes, little-endian bits.
    """
    name = "bitpack"

    # ---- encode ----
    def encode(self, arr: np.ndarray) -> Encoded:
        a = np.asarray(arr)
        v, dt = _lanes_of(a)
        n = v.size
        n_blocks = (n + BLOCK - 1) // BLOCK
        if n == 0:
            return Encoded(codec=self.name, dtype=dt.str, shape=tuple(a.shape), nbytes=0, raw_bytes=0,
                           payload={"table": np.zeros(0, np.uint32), "words": np.zeros(0, np.uint32)},
                           meta={"n": 0, "n_blocks": 0, "delta_blocks": 0, "mean_width": 0.0})
        counts = np.full(n_blocks, BLOCK, np.int64)
        counts[-1] = n - (n_blocks - 1) * BLOCK
        padded = np.zeros(n_blocks * BLOCK, np.int64)
        padded[:n] = v
        X = padded.reshape(n_blocks, BLOCK)
        lane = np.arange(BLOCK)[None, :]
        live = lane < counts[:, None]
        # frame of reference: value - block minimum
        lo = np.where(live, X, np.iinfo(np.int64).max).min(axis=1)
        hi = np.where(live, X, np.iinfo(np.int64).min).max(axis=1)
        w_for = _widths(hi - lo)
        # zigzag delta: lane 0 carries nothing (the base is in the table)
        d = np.diff(X, axis=1, prepend=X[:, :1])
        z = np.where(d >= 0, d * 2, -d * 2 - 1)
        z[:, 0] = 0
        z = np.where(live, z, 0)
        zmax = z.max(axis=1)
        w_delta = _widths(zmax)
        use_delta = (zmax < 2 ** 32) & (w_delta < w_for) & (counts > 1)
        widths = np.where(use_delta, w_delta, w_for).astype(np.int64)
        base = np.where(use_delta, X[:, 0], lo)
        vals = np.where(use_delta[:, None], z, X - lo[:, None])
        vals = np.where(live, vals, 0)
        words, offsets = _pack_blocks(vals, widths, counts)
        table = np.empty((n_blocks, 4), np.uint32)
        table[:, 0] = offsets
        table[:, 1] = base & 0xFFFFFFFF
        table[:, 2] = (use_delta.astype(np.int64) << 8) | widths
        table[:, 3] = counts
        return Encoded(codec=self.name, dtype=dt.str, shape=tuple(a.shape),
                       nbytes=int(table.nbytes + words.nbytes), raw_bytes=int(a.nbytes),
                       payload={"table": table.reshape(-1), "words": words},
                       meta={"n": n, "n_blocks": n_blocks, "delta_blocks": int(use_delta.sum()),
                             "mean_width": float((widths * counts).sum() / n)})

    # ---- decode on the host ----
    def decode(self, enc: Encoded) -> np.ndarray:
        if enc.on_device:
            raise ValueError("payload lives on the device; use decode_device() or read it back first")
        n = enc.n
        if n == 0:
            return _lanes_to_dtype(np.zeros(0, np.uint32), enc.dtype, enc.shape)
        table = np.asarray(enc.payload["table"], np.uint32).reshape(-1, 4).astype(np.int64)
        words = np.asarray(enc.payload["words"], np.uint32)
        offsets, base, mw, counts = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
        widths = mw & 0xFF
        delta = (mw >> 8) == _MODE_DELTA
        vals = _unpack_blocks(words, offsets, widths, counts)              # (n_blocks, BLOCK)
        # frame of reference
        lanes = vals + base[:, None]
        # zigzag delta: unzigzag, prefix-sum within the block, add the base
        if delta.any():
            z = vals[delta]
            d = np.where(z & 1, -((z + 1) >> 1), z >> 1)
            d[:, 0] = 0
            lanes[delta] = np.cumsum(d, axis=1) + base[delta][:, None]
        out = (lanes.reshape(-1)[:n] & 0xFFFFFFFF).astype(np.uint32)
        return _lanes_to_dtype(out, enc.dtype, enc.shape)

    # ---- device paths ----
    def upload(self, enc: Encoded, device) -> Encoded:
        """The payload moved to device buffers; the host keeps nothing. `nbytes` is unchanged."""
        if enc.on_device:
            return enc
        import wgpu
        usage = wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
        table = np.ascontiguousarray(np.asarray(enc.payload["table"], np.uint32))
        words = np.ascontiguousarray(np.asarray(enc.payload["words"], np.uint32))
        payload = {"table": device.create_buffer_with_data(data=_pad4(table), usage=usage),
                   "words": device.create_buffer_with_data(data=_pad4(words), usage=usage)}
        return Encoded(codec=enc.codec, dtype=enc.dtype, shape=enc.shape, nbytes=enc.nbytes, raw_bytes=enc.raw_bytes,
                       payload=payload, meta=dict(enc.meta), on_device=True)

    def decode_device(self, enc: Encoded, device, out_buffer=None):
        """Decode into a device buffer of u32 lanes (one per element) and return the buffer."""
        import wgpu
        n, n_blocks = enc.n, int(enc.meta["n_blocks"])
        out_bytes = max(4, 4 * n)
        if out_buffer is None:
            out_buffer = device.create_buffer(size=out_bytes, usage=wgpu.BufferUsage.STORAGE |
                                              wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        if n == 0:
            return out_buffer
        up = enc if enc.on_device else self.upload(enc, device)
        pipe, layout = _pipeline(device)
        params = device.create_buffer_with_data(data=np.array([n, n_blocks, 0, 0], np.uint32),
                                                usage=wgpu.BufferUsage.UNIFORM)
        bg = device.create_bind_group(layout=layout, entries=[
            {"binding": 0, "resource": {"buffer": up.payload["table"], "offset": 0, "size": up.payload["table"].size}},
            {"binding": 1, "resource": {"buffer": up.payload["words"], "offset": 0, "size": up.payload["words"].size}},
            {"binding": 2, "resource": {"buffer": out_buffer, "offset": 0, "size": out_buffer.size}},
            {"binding": 3, "resource": {"buffer": params, "offset": 0, "size": params.size}},
        ])
        enc_ = device.create_command_encoder()
        cp = enc_.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((n_blocks + 63) // 64)
        cp.end()
        device.queue.submit([enc_.finish()])
        return out_buffer


def _pad4(a: np.ndarray) -> np.ndarray:
    """wgpu buffers need a non-zero, 4-byte-aligned size."""
    return a if a.nbytes >= 4 else np.zeros(1, np.uint32)


#: One thread per block. Lanes are read from the packed word stream with a two-word window,
#: then either rebased (frame of reference) or prefix-summed from the base (zigzag delta).
WGSL_BITPACK_DECODE = """
struct Params { n: u32, n_blocks: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> table: array<u32>;
@group(0) @binding(1) var<storage, read> words: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn lane(off: u32, width: u32, i: u32) -> u32 {
    let start = i * width;
    let wi = off + (start >> 5u);
    let bo = start & 31u;
    var v: u32 = words[wi] >> bo;
    if (bo > 0u && bo + width > 32u) {
        v = v | (words[wi + 1u] << (32u - bo));
    }
    if (width < 32u) {
        v = v & ((1u << width) - 1u);
    }
    return v;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.n_blocks) { return; }
    let off = table[b * 4u];
    let base = table[b * 4u + 1u];
    let mw = table[b * 4u + 2u];
    let cnt = table[b * 4u + 3u];
    let mode = mw >> 8u;
    let width = mw & 0xFFu;
    let first = b * 128u;
    if (mode == 0u) {
        for (var i: u32 = 0u; i < cnt; i = i + 1u) {
            var v: u32 = 0u;
            if (width > 0u) { v = lane(off, width, i); }
            out[first + i] = base + v;
        }
    } else {
        var acc: u32 = base;
        out[first] = acc;
        for (var i: u32 = 1u; i < cnt; i = i + 1u) {
            var z: u32 = 0u;
            if (width > 0u) { z = lane(off, width, i); }
            var d: u32 = z >> 1u;
            if ((z & 1u) == 1u) { d = 0u - d - 1u; }
            acc = acc + d;
            out[first + i] = acc;
        }
    }
}
"""

_PIPELINES: dict = {}


def _pipeline(device):
    key = id(device)
    hit = _PIPELINES.get(key)
    if hit is not None:
        return hit
    import wgpu
    module = device.create_shader_module(code=WGSL_BITPACK_DECODE)
    vis = wgpu.ShaderStage.COMPUTE
    layout = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": vis, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pipe = device.create_compute_pipeline(layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
                                          compute={"module": module, "entry_point": "main"})
    _PIPELINES[key] = (pipe, layout)
    return pipe, layout


# ---------------------------------------------------------------------------
# Bytes codec: everything that is not an integer
# ---------------------------------------------------------------------------

def _backends() -> dict:
    out = {}
    try:
        import zstandard
        out["zstd"] = (lambda b: zstandard.ZstdCompressor(level=3).compress(b),
                       lambda b: zstandard.ZstdDecompressor().decompress(b))
    except ImportError:
        pass
    try:
        import lz4.frame
        out["lz4"] = (lz4.frame.compress, lz4.frame.decompress)
    except ImportError:
        pass
    out["zlib"] = (lambda b: zlib.compress(b, 6), zlib.decompress)
    return out


class BytesCodec:
    """zstd, lz4 or zlib over the raw bytes — whichever is installed, or the one you name.
    Not GPU-decodable; floats, strings and anything else that is not an integer go here."""

    def __init__(self, backend: Optional[str] = None):
        avail = _backends()
        if backend is None:
            backend = next(iter(avail))
        if backend not in avail:
            raise ValueError(f"no such backend {backend!r}; available: {list(avail)}")
        self.backend = backend
        self._enc, self._dec = avail[backend]
        self.name = f"bytes-{backend}"

    def encode(self, arr: np.ndarray) -> Encoded:
        a = np.ascontiguousarray(np.asarray(arr))
        blob = self._enc(a.tobytes())
        return Encoded(codec=self.name, dtype=a.dtype.str, shape=tuple(a.shape), nbytes=len(blob),
                       raw_bytes=int(a.nbytes), payload={"blob": blob}, meta={"n": int(a.size)})

    def decode(self, enc: Encoded) -> np.ndarray:
        if enc.on_device:
            raise ValueError("payload lives on the device; read it back first")
        raw = self._dec(enc.payload["blob"])
        return np.frombuffer(raw, dtype=np.dtype(enc.dtype)).reshape(enc.shape).copy()


def bitpack() -> BitpackCodec:
    return BitpackCodec()


def bytes_codec(backend: Optional[str] = None) -> BytesCodec:
    return BytesCodec(backend)


# ---------------------------------------------------------------------------
# Sparse matrices: three arrays, each packed on its own
# ---------------------------------------------------------------------------

def _integer_valued(a: np.ndarray) -> bool:
    if a.dtype.kind in "iu":
        return True
    if a.dtype.kind != "f" or a.size == 0:
        return a.size == 0 and a.dtype.kind == "f"
    if not np.isfinite(a).all():
        return False
    r = np.rint(a)
    return bool(np.array_equal(r, a) and abs(r).max() <= 2 ** 31 - 1)


def _encode_any(arr: np.ndarray, int_codec, byte_codec) -> Encoded:
    """Integers (or integer-valued floats) through the packer, the rest through bytes."""
    a = np.asarray(arr)
    if a.dtype.kind in "iu":
        try:
            return int_codec.encode(a)
        except ValueError:
            return byte_codec.encode(a)
    if a.dtype.kind == "f" and _integer_valued(a):
        e = int_codec.encode(a.astype(np.int32))
        e.meta["float_dtype"] = a.dtype.str
        e.raw_bytes = int(a.nbytes)
        return e
    return byte_codec.encode(a)


def _decode_any(enc: Encoded, int_codec, byte_codec) -> np.ndarray:
    if enc.codec == "bytes-raw":                      # already decoded on a device and read back
        out = np.asarray(enc.payload["array"])
    elif enc.codec == int_codec.name:
        out = int_codec.decode(enc)
    else:
        out = byte_codec.decode(enc)
    fd = enc.meta.get("float_dtype")
    return out.astype(fd) if fd and out.dtype.str != fd else out


def encode_csr(X, int_codec=None, byte_codec=None) -> dict:
    """A CSR/CSC matrix as row lengths + indices + data, each packed on its own."""
    import scipy.sparse as sp
    int_codec = int_codec or BitpackCodec()
    byte_codec = byte_codec or BytesCodec()
    fmt = "csc" if sp.isspmatrix_csc(X) else "csr"
    M = X if sp.isspmatrix_csc(X) or sp.isspmatrix_csr(X) else sp.csr_matrix(X)
    lengths = np.diff(M.indptr).astype(np.int32)
    parts = {"lengths": _encode_any(lengths, int_codec, byte_codec),
             "indices": _encode_any(M.indices, int_codec, byte_codec),
             "data": _encode_any(M.data, int_codec, byte_codec)}
    raw = int(M.indptr.nbytes + M.indices.nbytes + M.data.nbytes)
    return {"kind": "csr", "format": fmt, "shape": tuple(M.shape), "dtype": M.dtype.str,
            "index_dtype": M.indices.dtype.str, "parts": parts,
            "nbytes": sum(p.nbytes for p in parts.values()), "raw_bytes": raw}


def decode_csr(pack: dict, int_codec=None, byte_codec=None):
    import scipy.sparse as sp
    int_codec = int_codec or BitpackCodec()
    byte_codec = byte_codec or BytesCodec()
    lengths = _decode_any(pack["parts"]["lengths"], int_codec, byte_codec)
    indices = _decode_any(pack["parts"]["indices"], int_codec, byte_codec).astype(pack.get("index_dtype", np.int32))
    data = _decode_any(pack["parts"]["data"], int_codec, byte_codec).astype(pack["dtype"])
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(pack.get("index_dtype", np.int32))
    cls = sp.csc_matrix if pack["format"] == "csc" else sp.csr_matrix
    return cls((data, indices, indptr), shape=tuple(pack["shape"]))


def ratio_report(arrays: dict, int_codec=None, byte_codec=None) -> dict:
    """What each array would pack to: {name: {codec, raw, packed, ratio, mean_width}}."""
    int_codec = int_codec or BitpackCodec()
    byte_codec = byte_codec or BytesCodec()
    out = {}
    for name, a in arrays.items():
        e = _encode_any(np.asarray(a), int_codec, byte_codec)
        out[name] = {"codec": e.codec, "raw": e.raw_bytes, "packed": e.nbytes, "ratio": e.ratio,
                     "mean_width": e.meta.get("mean_width")}
    return out


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

@dataclass
class StoreEntry:
    name: str
    kind: str                         # "array" | "csr"
    codec: str
    packed_bytes: int
    raw_bytes: int
    dtype: str
    shape: tuple
    on_device: bool = False
    enc: Any = None                   # Encoded, or the csr pack dict
    lease: Any = None

    @property
    def ratio(self) -> float:
        return self.raw_bytes / max(self.packed_bytes, 1)

    def __repr__(self) -> str:
        return (f"StoreEntry({self.name}, {self.kind}, {self.codec}, {human(self.raw_bytes)} → "
                f"{human(self.packed_bytes)}, x{self.ratio:.1f}{', device' if self.on_device else ''})")


class DeviceArray:
    """A decoded block living in a device buffer: use it in place, or `read()` it back.

    `layout` says what the buffer holds: ``"lanes-u32"`` (one u32 per element, the bitpack
    kernel's output; signed and narrow types are two's-complement lanes) or ``"native"`` (the
    array's own dtype, byte for byte — what a plug-in codec that decodes floats produces).
    """

    def __init__(self, buffer, dtype, shape, n: int, device, layout: str = "lanes-u32"):
        self.buffer, self.dtype, self.shape, self.n, self.device = buffer, dtype, shape, n, device
        self.layout = layout

    def read(self) -> np.ndarray:
        raw = self.device.queue.read_buffer(self.buffer)
        if self.layout == "native":
            dt = np.dtype(self.dtype)
            return np.frombuffer(raw, dt)[: self.n].reshape(self.shape).copy()
        words = np.frombuffer(raw, np.uint32)[: self.n]
        return _lanes_to_dtype(words, self.dtype, self.shape)

    def __repr__(self) -> str:
        return f"DeviceArray({self.dtype}{list(self.shape)}, {human(4 * self.n)} on device)"


class _Held:
    """A decoded block plus the lease that pays for it: a context manager, or `.value` and
    `.release()` by hand."""

    def __init__(self, value, lease):
        self.value, self._lease = value, lease

    def release(self) -> None:
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        self.release()
        return False

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass


class CompressedStore:
    """Blocks resident packed, charged to a budget tier at their packed size.

    ::

        st = g.compressed_store(budget, "ram")
        st.put("blk0", X)                 # leases the packed bytes
        with st.get("blk0") as X:         # raw bytes leased only while held
            ...
        st.print_report()

    With `tier="vram"` and a wgpu device the payload lives in device buffers; `get_device()`
    decodes on the GPU into a buffer the caller uses in place.
    """

    def __init__(self, budget: Budget, tier: str = "ram", codec=None, device=None):
        if tier == "vram" and device is None:
            raise ValueError("tier='vram' needs a wgpu device to hold the blocks")
        self.budget, self.tier, self.device = budget, tier, device
        self.int_codec = get_codec(codec) if codec is not None else BitpackCodec()
        self.byte_codec = BytesCodec()
        self.entries: dict[str, StoreEntry] = {}

    # ---- leases ----
    def _lease(self, nbytes: int):
        from .workspace import try_acquire
        nbytes = int(nbytes)
        if nbytes == 0:
            return None
        if nbytes > self.budget.stats()[self.tier]["capacity"]:
            raise MemoryError(f"cannot fit {human(nbytes)} in the '{self.tier}' tier "
                              f"(capacity {self.budget.stats()[self.tier]['capacity_human']})")
        lease = try_acquire(self.budget, self.tier, nbytes)
        if lease is None:
            raise MemoryError(f"{human(nbytes)} does not fit in the '{self.tier}' tier right now "
                              f"({human(self.budget.available(self.tier))} free); evict something first")
        return lease

    # ---- writing ----
    def put(self, name: str, obj, overwrite: bool = False) -> StoreEntry:
        import scipy.sparse as sp
        if name in self.entries:
            if not overwrite:
                raise KeyError(f"block {name!r} is already stored; pass overwrite=True")
            self.evict(name)
        if sp.issparse(obj):
            pack = encode_csr(obj, self.int_codec, self.byte_codec)
            if self.device is not None and self.tier == "vram" and hasattr(self.int_codec, "upload"):
                for k, e in pack["parts"].items():
                    if e.codec == self.int_codec.name:
                        pack["parts"][k] = self.int_codec.upload(e, self.device)
            entry = StoreEntry(name, "csr", self.int_codec.name, int(pack["nbytes"]), int(pack["raw_bytes"]),
                               pack["dtype"], tuple(pack["shape"]), on_device=(self.tier == "vram"), enc=pack)
        elif _is_table(obj):
            return self.put_table(name, obj)
        else:
            e = _encode_any(np.asarray(obj), self.int_codec, self.byte_codec)
            if self.device is not None and self.tier == "vram" and e.codec == self.int_codec.name \
                    and hasattr(self.int_codec, "upload"):
                e = self.int_codec.upload(e, self.device)
            entry = StoreEntry(name, "array", e.codec, int(e.nbytes), int(e.raw_bytes), e.dtype, tuple(e.shape),
                               on_device=e.on_device, enc=e)
        entry.lease = self._lease(entry.packed_bytes)
        self.entries[name] = entry
        return entry

    # ---- reading ----
    def _decoded(self, entry: StoreEntry):
        if entry.kind == "table":
            return self._decoded_table(entry)
        if entry.kind == "csr":
            pack = entry.enc
            parts = dict(pack["parts"])
            if any(e.on_device for e in parts.values()):
                parts = {k: (_readback(self.int_codec, e, self.device) if e.on_device else e)
                         for k, e in parts.items()}
            return decode_csr({**pack, "parts": parts}, self.int_codec, self.byte_codec)
        e = entry.enc
        if e.on_device:
            e = _readback(self.int_codec, e, self.device)
        return _decode_any(e, self.int_codec, self.byte_codec)

    def get(self, name: str) -> "_Held":
        """The decoded block, with its raw bytes leased from the tier while you hold it. The
        lease is taken (and refused with MemoryError) at the call, not at `with`."""
        entry = self.entry(name)
        lease = self._lease(entry.raw_bytes)
        try:
            value = self._decoded(entry)
        except Exception:
            if lease is not None:
                lease.release()
            raise
        return _Held(value, lease)

    def get_device(self, name: str) -> "_Held":
        """Decode on the device into a buffer you use in place (vram leased for the raw bytes)."""
        entry = self.entry(name)
        if self.device is None:
            raise ValueError("get_device() needs a store opened with a wgpu device")
        if entry.kind == "csr":
            raise ValueError("a sparse block decodes to three arrays; use get() for sparse blocks")
        e = entry.enc
        if e.codec != self.int_codec.name:
            raise ValueError(f"block {name!r} is {e.codec}, which has no device decoder; use get()")
        if not hasattr(self.int_codec, "decode_device"):
            raise ValueError(f"codec {self.int_codec.name!r} has no device decoder; use get()")
        layout = e.meta.get("device_layout", "lanes-u32")
        out_bytes = max(4 * e.n, 4) if layout == "lanes-u32" else max(int(e.raw_bytes), 4)
        lease = self._lease(out_bytes)
        try:
            buf = self.int_codec.decode_device(e, self.device)
        except Exception:
            if lease is not None:
                lease.release()
            raise
        return _Held(DeviceArray(buf, e.dtype, e.shape, e.n, self.device, layout), lease)

    # ---- tables: one block per column, DuckDB- and Polars-ready ----
    def put_table(self, name: str, table, overwrite: bool = False) -> StoreEntry:
        """A pyarrow Table / RecordBatch, or a dict of columns, packed column by column.
        Integers go through the packer, floats and strings through bytes."""
        import pyarrow as pa
        if name in self.entries:
            if not overwrite:
                raise KeyError(f"block {name!r} is already stored; pass overwrite=True")
            self.evict(name)
        t = table if isinstance(table, pa.Table) else (
            pa.Table.from_batches([table]) if isinstance(table, pa.RecordBatch) else pa.table(table))
        cols: dict[str, Any] = {}
        raw = packed = 0
        for field_ in t.schema:
            col = t.column(field_.name).combine_chunks()
            if pa.types.is_integer(field_.type) or pa.types.is_floating(field_.type) or pa.types.is_boolean(field_.type):
                a = col.to_numpy(zero_copy_only=False)
                if col.null_count:
                    raise ValueError(f"column {field_.name!r} has nulls; fill them before packing")
                e = _encode_any(a.astype(np.uint8) if a.dtype == bool else a, self.int_codec, self.byte_codec)
                if a.dtype == bool:
                    e.meta["bool"] = True
                if self.device is not None and self.tier == "vram" and e.codec == self.int_codec.name \
                        and hasattr(self.int_codec, "upload"):
                    e = self.int_codec.upload(e, self.device)
                cols[field_.name] = ("array", e, str(field_.type))
            else:                                                  # strings, lists, timestamps: Arrow IPC bytes
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, pa.schema([field_])) as w:
                    w.write_table(pa.table({field_.name: col}))
                blob = sink.getvalue().to_pybytes()
                e = self.byte_codec.encode(np.frombuffer(blob, np.uint8))
                e.meta["arrow_ipc"] = True
                cols[field_.name] = ("ipc", e, str(field_.type))
            raw += e.raw_bytes
            packed += e.nbytes
        entry = StoreEntry(name, "table", self.int_codec.name, int(packed), int(raw), "table",
                           (t.num_rows, t.num_columns), on_device=(self.tier == "vram"),
                           enc={"columns": cols, "schema": t.schema.serialize().to_pybytes(), "n_rows": t.num_rows})
        entry.lease = self._lease(entry.packed_bytes)
        self.entries[name] = entry
        return entry

    def _decoded_table(self, entry: StoreEntry):
        import pyarrow as pa
        schema = pa.ipc.read_schema(pa.py_buffer(entry.enc["schema"]))
        arrays = []
        for field_ in schema:
            kind, e, _ = entry.enc["columns"][field_.name]
            if kind == "ipc":
                blob = self.byte_codec.decode(e).tobytes()
                with pa.ipc.open_stream(pa.py_buffer(blob)) as r:
                    arrays.append(r.read_all().column(0).combine_chunks())
            else:
                if e.on_device:
                    e = _readback(self.int_codec, e, self.device)
                a = _decode_any(e, self.int_codec, self.byte_codec)
                if e.meta.get("bool"):
                    a = a.astype(bool)
                arrays.append(pa.array(a, type=field_.type))
        return pa.Table.from_arrays(arrays, schema=schema)

    def get_table(self, name: str) -> "_Held":
        """A packed table back as a pyarrow Table (raw bytes leased while held)."""
        entry = self.entry(name)
        if entry.kind != "table":
            raise ValueError(f"block {name!r} is a {entry.kind}, not a table")
        return self.get(name)

    def duckdb(self, con, names: Optional[Sequence[str]] = None) -> dict:
        """Register packed tables with a DuckDB connection as views named after their blocks.
        Each is decoded once, on registration, and its raw bytes stay leased until `release()`
        on the returned holds (or until the store is cleared)."""
        holds = {}
        for name in (names or [n for n, e in self.entries.items() if e.kind == "table"]):
            h = self.get_table(name)
            con.register(name, h.value)
            holds[name] = h
        return holds

    # ---- bookkeeping ----
    def entry(self, name: str) -> StoreEntry:
        if name not in self.entries:
            raise KeyError(f"no block {name!r}; have {list(self.entries)}")
        return self.entries[name]

    def evict(self, name: str) -> None:
        entry = self.entries.pop(name)
        if entry.lease is not None:
            entry.lease.release()
            entry.lease = None

    def clear(self) -> None:
        for name in list(self.entries):
            self.evict(name)

    def report(self) -> dict:
        raw = sum(e.raw_bytes for e in self.entries.values())
        packed = sum(e.packed_bytes for e in self.entries.values())
        cap = int(self.budget.stats()[self.tier]["capacity"])
        ratio = raw / max(packed, 1) if packed else 1.0
        return {"tier": self.tier, "codec": self.int_codec.name, "blocks": len(self.entries), "raw": raw,
                "packed": packed, "ratio": ratio, "capacity": cap, "effective_capacity": int(cap * ratio),
                "raw_human": human(raw), "packed_human": human(packed), "capacity_human": human(cap),
                "effective_capacity_human": human(int(cap * ratio))}

    def print_report(self) -> None:
        r = self.report()
        print(f"compressed store ({r['tier']}, {r['codec']}): {r['blocks']} block(s), {r['raw_human']} "
              f"represented in {r['packed_human']} ({r['ratio']:.1f}x); tier capacity {r['capacity_human']} "
              f"now holds ~{r['effective_capacity_human']} of data")

    def __contains__(self, name) -> bool:
        return name in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries.values())

    def __repr__(self) -> str:
        r = self.report()
        return f"CompressedStore({r['tier']}, {r['blocks']} blocks, {r['raw_human']} in {r['packed_human']})"


def _is_table(obj) -> bool:
    try:
        import pyarrow as pa
    except ImportError:
        return False
    return isinstance(obj, (pa.Table, pa.RecordBatch)) or (
        isinstance(obj, dict) and obj and all(isinstance(v, (np.ndarray, list, pa.Array, pa.ChunkedArray))
                                              for v in obj.values()))


def _readback(codec, enc: Encoded, device) -> Encoded:
    """A device-resident payload decoded on the device and brought back as a host Encoded of
    the *decoded* array — an identity-coded array the host side can finish."""
    buf = codec.decode_device(enc, device)
    da = DeviceArray(buf, enc.dtype, enc.shape, enc.n, device, enc.meta.get("device_layout", "lanes-u32"))
    arr = da.read()
    host = Encoded(codec="bytes-raw", dtype=enc.dtype, shape=enc.shape, nbytes=enc.nbytes, raw_bytes=enc.raw_bytes,
                   payload={"array": arr}, meta=dict(enc.meta))
    return host


# ---------------------------------------------------------------------------
# Plugging in another codec
# ---------------------------------------------------------------------------

class CodecError(AssertionError):
    """A codec failed the conformance check; the message says which case and how."""


_REGISTRY: dict[str, Callable[[], Any]] = {"bitpack": BitpackCodec, "bytes": BytesCodec}


def register_codec(name: str, factory: Callable[[], Any]) -> None:
    """Make `get_codec(name)` return `factory()`. A package can also advertise one through the
    ``geminga.codecs`` entry-point group and skip this call."""
    _REGISTRY[name] = factory


def available_codecs() -> list:
    """Registered names plus any ``geminga.codecs`` entry points installed on this machine."""
    names = set(_REGISTRY)
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="geminga.codecs") if hasattr(eps, "select") else eps.get("geminga.codecs", [])
        names |= {ep.name for ep in group}
    except Exception:
        pass
    return sorted(names)


def get_codec(name_or_codec):
    """A codec instance from a name (registry, then entry points) or the object itself."""
    if not isinstance(name_or_codec, str):
        c = name_or_codec
        if not (hasattr(c, "encode") and hasattr(c, "decode") and hasattr(c, "name")):
            raise TypeError(f"{c!r} does not look like a codec (needs .name, .encode, .decode)")
        return c
    name = name_or_codec
    if name in _REGISTRY:
        return _REGISTRY[name]()
    try:
        from importlib.metadata import entry_points
        eps = entry_points()
        group = eps.select(group="geminga.codecs") if hasattr(eps, "select") else eps.get("geminga.codecs", [])
        for ep in group:
            if ep.name == name:
                obj = ep.load()
                codec_ = obj() if isinstance(obj, type) else obj
                _REGISTRY[name] = (lambda o=codec_: o)
                return codec_
    except Exception as e:                      # a broken plug-in must say so, not look absent
        raise ImportError(f"loading codec {name!r} from its entry point failed: {e}") from e
    raise KeyError(f"no codec {name!r}; available: {available_codecs()}")


def _conformance_cases(rng) -> list:
    return [
        ("counts int32", rng.poisson(2.0, 100_003).astype(np.int32) + 1),
        ("sorted int32", np.sort(rng.integers(0, 30000, 50_000)).astype(np.int32)),
        ("negative int32", rng.integers(-1000, 1000, 10_001).astype(np.int32)),
        ("int32 extremes", np.array([0, 2 ** 31 - 1, -(2 ** 31), 5, -1, 1], np.int32)),
        ("uint32 extremes", np.array([0, 2 ** 32 - 1, 7, 2 ** 31], np.uint32)),
        ("int16", rng.integers(-300, 300, 999).astype(np.int16)),
        ("uint8", rng.integers(0, 255, 1000).astype(np.uint8)),
        ("int64 small", rng.integers(0, 1 << 20, 3000).astype(np.int64)),
        ("single", np.array([42], np.int32)),
        ("empty", np.zeros(0, np.int32)),
        ("2d int32", rng.integers(0, 50, (64, 33)).astype(np.int32)),
        ("float32", rng.standard_normal(10_000).astype(np.float32)),
        ("float64", rng.standard_normal(1_000)),
        ("integer-valued float32", (np.round(rng.exponential(2, 5000)) + 1).astype(np.float32)),
    ]


def check_codec(codec, device=None, strict: bool = True, seed: int = 0) -> dict:
    """Run a codec through the cases the store relies on and report what it handled.

    For every case: `encode` must return an `Encoded` whose `raw_bytes` is the array's size and
    whose `dtype`/`shape` match; `decode` must give the array back exactly (a codec may refuse
    a dtype with `TypeError`/`ValueError`, which counts as *declined*, not failed — the store
    falls back to bytes for those). With a wgpu `device`, `decode_device` (when present) must
    agree with the host decoder bit for bit, in whichever `device_layout` the codec declares,
    and `upload` (when present) must leave `nbytes` unchanged and make `on_device` true.

    Returns ``{"codec", "passed": [...], "declined": [...], "failed": [...], "device": bool,
    "ratios": {case: ratio}}``. With `strict`, any failure raises :class:`CodecError`.
    """
    codec = get_codec(codec)
    rng = np.random.default_rng(seed)
    rep: dict = {"codec": codec.name, "passed": [], "declined": [], "failed": [], "device": False, "ratios": {}}
    problems = []
    has_dev = device is not None and hasattr(codec, "decode_device")
    for label, arr in _conformance_cases(rng):
        try:
            enc = codec.encode(arr)
        except (TypeError, ValueError) as e:
            rep["declined"].append(f"{label}: {e}")
            continue
        try:
            if not isinstance(enc, Encoded):
                raise CodecError(f"{label}: encode() returned {type(enc).__name__}, not Encoded")
            if enc.raw_bytes != arr.nbytes:
                raise CodecError(f"{label}: raw_bytes {enc.raw_bytes} != array size {arr.nbytes}")
            if np.dtype(enc.dtype) != arr.dtype or tuple(enc.shape) != arr.shape:
                raise CodecError(f"{label}: dtype/shape recorded as {enc.dtype}{enc.shape}, array is "
                                 f"{arr.dtype.str}{arr.shape}")
            if enc.nbytes < 0:
                raise CodecError(f"{label}: negative nbytes")
            out = codec.decode(enc)
            if out.dtype != arr.dtype or out.shape != arr.shape or not np.array_equal(out, arr):
                raise CodecError(f"{label}: decode() is not the inverse of encode()")
            rep["ratios"][label] = enc.ratio
            if has_dev:
                layout = enc.meta.get("device_layout", "lanes-u32")
                buf = codec.decode_device(enc, device)
                got = DeviceArray(buf, enc.dtype, enc.shape, enc.n, device, layout).read()
                if not np.array_equal(got, arr):
                    raise CodecError(f"{label}: decode_device() disagrees with decode() ({layout})")
                if hasattr(codec, "upload"):
                    up = codec.upload(enc, device)
                    if not up.on_device or up.nbytes != enc.nbytes:
                        raise CodecError(f"{label}: upload() must keep nbytes and set on_device")
                    buf2 = codec.decode_device(up, device)
                    got2 = DeviceArray(buf2, enc.dtype, enc.shape, enc.n, device, layout).read()
                    if not np.array_equal(got2, arr):
                        raise CodecError(f"{label}: decode_device() from an uploaded payload disagrees")
                rep["device"] = True
            rep["passed"].append(label)
        except CodecError as e:
            problems.append(str(e))
            rep["failed"].append(str(e))
        except Exception as e:
            problems.append(f"{label}: {type(e).__name__}: {e}")
            rep["failed"].append(f"{label}: {type(e).__name__}: {e}")
    if strict and problems:
        raise CodecError(f"codec {codec.name!r} failed {len(problems)} case(s):\n  " + "\n  ".join(problems))
    return rep


def compressed_store(budget: Budget, tier: str = "ram", codec=None, device=None) -> CompressedStore:
    """Blocks resident packed, charged to a budget tier at their packed size."""
    return CompressedStore(budget, tier, codec, device)

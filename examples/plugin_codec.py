"""A third-party codec, plugged into GEMINGA's compressed store — the template.

    python examples/plugin_codec.py

Everything a compressor needs to provide is here: a `name`, `encode()` returning an
`Encoded`, `decode()` giving the array back, and — optionally — `upload()`/`decode_device()`
for a GPU path. `g.check_codec()` is the conformance test; pass it and the store, the
reservation client and every reader that stores blocks will use the codec unchanged.

The example codec is run-length encoding, which is deliberately simple and deliberately
good at exactly one thing (long runs — sorted bin coordinates, zero-heavy columns) so that the
ratio table at the end shows where it beats the built-in packer and where it does not.

To ship it as a package, declare the entry point and GEMINGA finds it by name:

    [project.entry-points."geminga.codecs"]
    rle = "mypackage.codec:RunLengthCodec"

    store = g.compressed_store(budget, codec="rle")
"""
import numpy as np

import geminga as g
from geminga.codec import Encoded, check_codec, register_codec


class RunLengthCodec:
    """Runs of equal 32-bit integers as (value, length) pairs. Lossless; refuses floats."""
    name = "rle"

    def encode(self, arr: np.ndarray) -> Encoded:
        a = np.asarray(arr)
        if a.dtype.kind not in "iu":
            raise TypeError("rle packs integers; floats go through BytesCodec")
        if a.dtype.itemsize > 4:
            raise ValueError("rle packs 32-bit integers")
        v = a.reshape(-1)
        if v.size == 0:
            runs = np.zeros((0, 2), np.uint32)
        else:
            change = np.flatnonzero(np.diff(v) != 0) + 1
            starts = np.concatenate([[0], change])
            lengths = np.diff(np.concatenate([starts, [v.size]]))
            lanes = v[starts].astype(np.int64) & 0xFFFFFFFF          # two's-complement u32 lanes
            runs = np.stack([lanes, lengths], axis=1).astype(np.uint32)
        return Encoded(codec=self.name, dtype=a.dtype.str, shape=tuple(a.shape), nbytes=int(runs.nbytes),
                       raw_bytes=int(a.nbytes), payload={"runs": runs}, meta={"n": int(v.size), "n_runs": len(runs)})

    def decode(self, enc: Encoded) -> np.ndarray:
        if enc.on_device:
            raise ValueError("payload lives on the device")
        runs = np.asarray(enc.payload["runs"], np.uint32).reshape(-1, 2)
        lanes = np.repeat(runs[:, 0], runs[:, 1]) if len(runs) else np.zeros(0, np.uint32)
        from geminga.codec import _lanes_to_dtype
        return _lanes_to_dtype(lanes, enc.dtype, enc.shape)

    # No upload()/decode_device(): the store keeps the payload on the host and get_device()
    # says so. Add them, run check_codec(codec, device=...), and the GPU path is live.


if __name__ == "__main__":
    register_codec("rle", RunLengthCodec)
    rep = check_codec("rle")
    print(f"rle: {len(rep['passed'])} cases passed, {len(rep['declined'])} declined "
          f"({', '.join(d.split(':')[0] for d in rep['declined'])})")

    rng = np.random.default_rng(0)
    cases = {
        "sorted bin x (many runs of 1)": np.sort(rng.integers(0, 26000, 1_000_000)).astype(np.int32),
        "bin x, 200 bins wide (runs ~5000)": np.repeat(np.arange(200, dtype=np.int32), 5000),
        "UMI counts": rng.poisson(2.0, 1_000_000).astype(np.int32) + 1,
        "sparse flags (97% zero)": (rng.random(1_000_000) < 0.03).astype(np.int32),
    }
    from geminga.codec import BitpackCodec
    bp = BitpackCodec()
    rle = RunLengthCodec()
    print(f"\n{'array':<36} {'bitpack':>9} {'rle':>9}")
    for name, a in cases.items():
        print(f"{name:<36} {bp.encode(a).ratio:8.1f}x {rle.encode(a).ratio:8.1f}x")

    b = g.Budget(ram="16MB")
    st = g.compressed_store(b, "ram", codec="rle")
    st.put("flags", cases["sparse flags (97% zero)"])
    st.print_report()

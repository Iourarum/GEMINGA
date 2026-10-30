"""How fast can a long journal be searched — grep, ripgrep, Python, DuckDB, the trigram index?

Writes a synthetic journal of N records shaped like real ones (about 330 bytes each: a GEF
dataset key, a window, a bin, two or three gene names, the usual tags), then times:

  * `grep -n -F -i`, `rg -n -F -i`       — the Unix way, on the file itself (matching lines printed)
  * a Python line scan                    — what `Journal.search` does with no tool at all
  * DuckDB `read_json_auto`               — SQL over the same file
  * the trigram index (`Journal.index()`) — build, sidecar size, then queries

for four kinds of query: a term in a handful of lines (a selection id), a gene in about 1 % of
records, a term in every record ("load"), and a two-character term the index cannot narrow.

    python examples/bench_journal.py 1000000 /tmp/journal-bench
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import geminga as g
from geminga.journal import Journal

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.getcwd(), "journal-bench")
os.makedirs(OUT, exist_ok=True)
PATH = os.path.join(OUT, "journal.jsonl")

GENES = [f"G{i:05d}" for i in range(20_000)] + ["GFAP", "SOX2", "OLIG2", "EGFR", "PTPRC", "MKI67"]
RARE = "sel-7f3a9c2e1b4d"           # appears in ~10 lines


def write(n: int) -> None:
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    datasets = [{"path": f"/data/project/chip{i:03d}.gef", "kind": "gef", "key": f"{rng.integers(1 << 62):016x}",
                 "fingerprint": {"size": int(rng.integers(1 << 30, 1 << 40)), "mtime": 1760000000, "version": 2}}
                for i in range(40)]
    rare_at = set(rng.choice(n, 10, replace=False).tolist())
    with open(PATH, "w", encoding="utf-8") as fh:
        for i in range(n):
            d = datasets[i % 40]
            x0 = int(rng.integers(0, 20000)); y0 = int(rng.integers(0, 20000))
            w = int(rng.integers(200, 4000))
            genes = [GENES[int(k)] for k in rng.integers(0, 20_000, 2)]
            if rng.random() < 0.01:
                genes.append("GFAP")
            rec = {"t": 1760000000 + i * 0.37, "op": "load" if i % 9 else "rasterize", "dataset": d,
                   "unit": "spots", "window": [x0, x0 + w, y0, y0 + w], "level": {"bin": int(rng.choice([10, 20, 50, 100]))},
                   "genes": genes, "units": int(rng.integers(0, 500_000)), "bytes": int(rng.integers(0, 50_000_000)),
                   "seconds": round(float(rng.random()), 3), "via": "explorer" if i % 3 else "",
                   "selection": RARE if i in rare_at else "", "note": "", "tags": [f"frame={i % 500}"],
                   "host": "lab-laptop", "pid": 4242, "session": "a1b2c3d4", "schema": "geminga.journal/1"}
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    print(f"wrote {n:,} records, {os.path.getsize(PATH) / 1e6:.0f} MB in {time.perf_counter() - t0:.1f} s")


def timed(fn, repeat=3):
    best = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, r


def tool_count(tool: str, pat: str) -> int:
    """Lines matched, *printed with their numbers* (`-n`), as the index returns them — a bare
    `-c` count would be unfair to a search that hands back every matching line."""
    p = subprocess.run([tool, "-n", "-F", "-i", pat, PATH], capture_output=True)
    return p.stdout.count(b"\n")


def py_count(pat: str) -> int:
    pat = pat.lower()
    n = 0
    with open(PATH, encoding="utf-8") as fh:
        for s in fh:
            if pat in s.lower():
                n += 1
    return n


def main():
    if not os.path.exists(PATH) or sum(1 for _ in open(PATH, "rb")) != N:
        write(N)
    size_mb = os.path.getsize(PATH) / 1e6
    j = Journal(PATH)
    if os.path.exists(j.index_path):
        os.remove(j.index_path)
    print(f"\njournal: {N:,} records, {size_mb:.0f} MB   (geminga {g.__version__})")

    t0 = time.perf_counter()
    idx = j.index()
    build = time.perf_counter() - t0
    if idx is None:
        print("no TrigramIndex in this core"); return
    print(f"trigram index: built in {build:.2f} s, {idx.n_blocks:,} blocks of {idx.block_lines} lines, "
          f"{idx.n_trigrams:,} trigrams, sidecar {os.path.getsize(j.index_path) / 1e6:.1f} MB "
          f"({100 * os.path.getsize(j.index_path) / os.path.getsize(PATH):.0f} % of the journal)")

    queries = [("rare id", RARE), ("gene in ~1 %", "GFAP"), ("every record", "load"), ("2 chars", "G1")]
    print(f"\n{'query':<14}{'hits':>9}  {'grep':>8} {'rg':>8} {'python':>8} {'trigram':>8}  blocks read")
    print("(grep/rg print matching lines with -n; trigram returns them to Python as (line, text))")
    for label, pat in queries:
        row = {}
        n_hits = None
        for tool in ("grep", "rg"):
            if shutil.which(tool):
                dt, n = timed(lambda t=tool: tool_count(t, pat))
                row[tool] = dt; n_hits = n
        dt, n = timed(lambda: py_count(pat), repeat=1)
        row["python"] = dt; n_hits = n
        dt, hits = timed(lambda: idx.search(pat, True))
        row["trigram"] = dt
        assert len(hits) == n_hits, (label, len(hits), n_hits)
        dt_j, hits_j = timed(lambda: j.grep(pat))
        row["journal"] = dt_j
        assert len(hits_j) == n_hits
        c = idx.candidates(pat)
        blocks = "all" if c is None else f"{len(c):,}/{idx.n_blocks:,}"
        print(f"{label:<14}{n_hits:>9,}  " + " ".join(f"{row.get(k, float('nan')) * 1000:>7.0f}ms" for k in ("grep", "rg", "python", "trigram"))
              + f"  {blocks}")

    # the structured query grep cannot do: a gene inside a window, through the index
    dt, res = timed(lambda: j.search("GFAP", overlaps=(5000, 5200, 5000, 5200)))
    print(f"\nJournal.search('GFAP', overlaps=(5000,5200,5000,5200)): {len(res)} records in {dt * 1000:.0f} ms (engine {j.engine})")
    try:
        import duckdb
        con = j.duckdb()
        dt, n = timed(lambda: con.execute(
            "SELECT count(*) FROM journal WHERE list_contains(genes, 'GFAP') "
            "AND \"window\"[1] < 5200 AND \"window\"[2] > 5000 AND \"window\"[3] < 5200 AND \"window\"[4] > 5000").fetchone()[0], repeat=2)
        print(f"DuckDB, same question over read_json_auto: {n} records in {dt:.2f} s (duckdb {duckdb.__version__})")
    except ImportError:
        pass

    # growth: append 1 % more and extend
    with open(PATH, "a", encoding="utf-8") as fh:
        for i in range(N // 100):
            fh.write(json.dumps({"t": 0, "op": "load", "dataset": {"key": "ffffffffffffffff", "path": "/x.gef", "kind": "gef"},
                                 "window": [0, 1, 0, 1], "genes": ["APPENDED"], "schema": "geminga.journal/1"},
                                separators=(",", ":")) + "\n")
    dt, hits = timed(lambda: idx.search("APPENDED", True), repeat=1)
    print(f"\nafter appending {N // 100:,} records: search before update {dt * 1000:.0f} ms (tail scanned), ", end="")
    t0 = time.perf_counter(); added = idx.update(); up = time.perf_counter() - t0
    dt, hits2 = timed(lambda: idx.search("APPENDED", True))
    print(f"update() indexed {added:,} lines in {up * 1000:.0f} ms, search after {dt * 1000:.1f} ms; {len(hits)} == {len(hits2)} hits")


if __name__ == "__main__":
    main()

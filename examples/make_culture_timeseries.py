"""A synthetic culture-monitoring time series: dissolved O2, pH and confluence every minute for
ten days, with four hypoxic episodes planted at known times. Not data — a test of whether a
time-range selection can find its own kind elsewhere in the series."""
import sys
import numpy as np
import polars as pl
out = sys.argv[1] if len(sys.argv) > 1 else "fixtures/culture.parquet"
rng = np.random.default_rng(11)
n = 10 * 24 * 60
t = np.arange(n, dtype=np.int64) * 60                       # seconds
day = t / 86400
o2 = 18 + 0.6 * np.sin(2 * np.pi * day) + rng.normal(0, 0.35, n)
ph = 7.35 - 0.02 * day + rng.normal(0, 0.012, n)
conf = 100 / (1 + np.exp(-(day - 5) * 0.9)) + rng.normal(0, 0.8, n)
EVENTS = [1.4, 3.9, 6.2, 8.7]                                # days
for d0 in EVENTS:
    s = (day >= d0) & (day < d0 + 0.25)                       # six-hour hypoxic episodes
    ramp = np.sin(np.pi * (day[s] - d0) / 0.25)
    o2[s] -= 11 * ramp
    ph[s] -= 0.09 * ramp
pl.DataFrame({"t": t, "o2_percent": o2, "ph": ph, "confluence": conf}).write_parquet(
    out, row_group_size=2048, statistics=True)
np.save(out.replace(".parquet", "_events.npy"), np.array(EVENTS))
print(f"{out}: {n:,} rows, hypoxic episodes at days {EVENTS}")

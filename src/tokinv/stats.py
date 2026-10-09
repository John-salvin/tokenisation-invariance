"""Statistics shared by every section: rank correlation, the dose-response
monotonicity statistic, its exact permutation p-value and its segment bootstrap."""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd

# Segment bootstrap used for every interval on the monotonicity statistic.
# The segment, not the bucket, is resampled, so a segment's buckets stay together.
BOOT_RESAMPLES = 300
BOOT_SEED = 0
MIN_BUCKET_N = 10        # a within-bucket correlation needs a real sample


def spearman(a, b) -> float:
    """Spearman's rho as the Pearson correlation of average ranks, NaN-safe.

    Returns NaN for fewer than three complete pairs or a constant input."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy() - (len(a) + 1) / 2
    rb = pd.Series(b).rank().to_numpy() - (len(b) + 1) / 2
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def dose_curve(df: pd.DataFrame, score: str = "score", human: str = "human",
               min_n: int = MIN_BUCKET_N) -> pd.DataFrame:
    """One row per non-canonical bucket: mean ratio and metric-human agreement."""
    rows = []
    for bucket, s in df[df.bucket != "canonical"].groupby("bucket", sort=True):
        if len(s) < min_n:
            continue
        rho = spearman(s[score], s[human])
        if not np.isnan(rho):
            rows.append({"bucket": bucket, "n": len(s),
                         "mean_ratio": float(s.ratio.mean()), "rho": rho})
    return pd.DataFrame(rows, columns=["bucket", "n", "mean_ratio", "rho"])


def monotonicity(df: pd.DataFrame, **kw) -> float:
    """Spearman between bucket fragmentation and bucket agreement (Section 4)."""
    c = dose_curve(df, **kw)
    return spearman(c.mean_ratio, c.rho) if len(c) >= 3 else float("nan")


@lru_cache(maxsize=None)
def _null_rhos(n: int) -> np.ndarray:
    """Spearman rho of every permutation of n distinct ranks (n <= 9)."""
    x = np.arange(n)
    d = np.array([((x - np.array(p)) ** 2).sum()
                  for p in itertools.permutations(range(n))])
    return 1 - 6 * d / (n * (n * n - 1))


def exact_p(rho: float, n: int) -> float:
    """Two-sided exact permutation p-value of Spearman's rho on n untied points."""
    null = _null_rhos(n)
    return float(np.mean(np.abs(null) >= abs(rho) - 1e-12))


def segment_bootstrap(df: pd.DataFrame, statistic, unit: str,
                      n_resamples: int = BOOT_RESAMPLES, seed: int = BOOT_SEED,
                      level: float = 0.95, return_draws: bool = False):
    """Percentile interval of ``statistic`` resampling whole units (segments).

    Each unit carries all its bucket rows, so the dependence between buckets
    computed from the same segments is preserved."""
    # positional rows of each unit, in original order; one iloc per resample
    # gives exactly the frame pd.concat of the per-unit frames would
    by = df.groupby(unit, sort=True).indices
    units = np.array(list(by))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_resamples):
        pick = rng.choice(units, len(units), replace=True)
        v = statistic(df.iloc[np.concatenate([by[u] for u in pick])].reset_index(drop=True))
        if not np.isnan(v):
            vals.append(v)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(vals, [a, 100 - a])
    if return_draws:
        return float(lo), float(hi), np.asarray(vals)
    return float(lo), float(hi)

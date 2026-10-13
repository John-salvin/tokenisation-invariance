"""Tests for two dependent correlations that share one variable: did the
metric's agreement with MQM change between two renderings of the SAME
segments? Same sentences, same MQM, so the correlations are dependent and a
CI-overlap comparison is wrong (it ignores the pairing)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float


def meng_rosenthal_rubin(r_jh: float, r_kh: float, r_jk: float, n: int) -> TestResult:
    """Meng, Rosenthal and Rubin (1992) z-test. j, k: the two predictors (e.g.
    canonical and re-segmented COMET), h: the shared outcome (MQM)."""
    for r in (r_jh, r_kh, r_jk):
        if not -1.0 <= r <= 1.0:
            raise ValueError(f"correlation out of range: {r}")
    if n < 4:
        raise ValueError("n must be >= 4")
    r_sq_mean = (r_jh ** 2 + r_kh ** 2) / 2.0
    denom = 1.0 - r_sq_mean
    f = min((1.0 - r_jk) / (2.0 * denom) if denom > 0 else 1.0, 1.0)
    h = (1.0 - f * r_sq_mean) / (1.0 - r_sq_mean) if r_sq_mean < 1 else 1.0
    z = (np.arctanh(r_jh) - np.arctanh(r_kh)) * np.sqrt((n - 3) / (2.0 * (1.0 - r_jk) * h))
    return TestResult("meng_rosenthal_rubin", float(z), float(2.0 * sps.norm.sf(abs(z))))


def steiger(r_jh: float, r_kh: float, r_jk: float, n: int) -> TestResult:
    """Steiger (1980) t-test, a cross-check on the same hypothesis."""
    if n < 4:
        raise ValueError("n must be >= 4")
    det = 1.0 - r_jh ** 2 - r_kh ** 2 - r_jk ** 2 + 2.0 * r_jh * r_kh * r_jk
    r_bar = (r_jh + r_kh) / 2.0
    t = ((r_jh - r_kh) * np.sqrt((n - 1) * (1.0 + r_jk))
         / np.sqrt(2.0 * (n - 1) / (n - 3) * det + r_bar ** 2 * (1.0 - r_jk) ** 3))
    return TestResult("steiger", float(t), float(2.0 * sps.t.sf(abs(t), df=n - 3)))


def paired_permutation_corr(x_a, x_b, y, n_rounds: int = 10_000, seed: int = 42) -> TestResult:
    """Paired permutation test (Deutsch, Dror and Roth 2021): under the null the
    two renderings are exchangeable per item, so each round swaps a random
    subset. Add-one smoothing, so p is never reported as 0."""
    x_a, x_b, y = map(np.asarray, (x_a, x_b, y))
    ok = ~(np.isnan(x_a) | np.isnan(x_b) | np.isnan(y))
    x_a, x_b, y = x_a[ok], x_b[ok], y[ok]
    n = len(y)
    rho = lambda u, v: sps.spearmanr(u, v)[0]
    observed = rho(x_a, y) - rho(x_b, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_rounds):
        swap = rng.random(n) < 0.5
        if abs(rho(np.where(swap, x_b, x_a), y) - rho(np.where(swap, x_a, x_b), y)) >= abs(observed):
            count += 1
    return TestResult("paired_permutation", float(observed), (count + 1) / (n_rounds + 1))

"""Statistical helpers: bootstrap intervals and permutation tests.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    detail: dict

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("detail"))
        return d


def meng_rosenthal_rubin(r_jh: float, r_kh: float, r_jk: float, n: int) -> TestResult:
    for nm, r in (("r_jh", r_jh), ("r_kh", r_kh), ("r_jk", r_jk)):
        if not -1.0 <= r <= 1.0:
            raise ValueError(f"{nm} out of range: {r}")
    if n < 4:
        raise ValueError("n must be >= 4")

    z_j = np.arctanh(r_jh)
    z_k = np.arctanh(r_kh)

    r_sq_mean = (r_jh ** 2 + r_kh ** 2) / 2.0
    denom = 1.0 - r_sq_mean
    f = (1.0 - r_jk) / (2.0 * denom) if denom > 0 else 1.0
    f = min(f, 1.0)
    h = (1.0 - f * r_sq_mean) / (1.0 - r_sq_mean) if r_sq_mean < 1 else 1.0

    z = (z_j - z_k) * np.sqrt((n - 3) / (2.0 * (1.0 - r_jk) * h))
    p = 2.0 * sps.norm.sf(abs(z))
    return TestResult("meng_rosenthal_rubin", float(z), float(p),
                      {"r_jh": r_jh, "r_kh": r_kh, "r_jk": r_jk, "n": n,
                       "delta_rho": r_jh - r_kh,
                       "delta_rho_squared": r_jh ** 2 - r_kh ** 2,
                       "f": float(f), "h": float(h)})


def steiger(r_jh: float, r_kh: float, r_jk: float, n: int) -> TestResult:
    if n < 4:
        raise ValueError("n must be >= 4")
    det = (1.0 - r_jh ** 2 - r_kh ** 2 - r_jk ** 2
           + 2.0 * r_jh * r_kh * r_jk)
    r_bar = (r_jh + r_kh) / 2.0
    num = (r_jh - r_kh) * np.sqrt((n - 1) * (1.0 + r_jk))
    den = np.sqrt(2.0 * (n - 1) / (n - 3) * det + (r_bar ** 2) * ((1.0 - r_jk) ** 3))
    t = num / den
    p = 2.0 * sps.t.sf(abs(t), df=n - 3)
    return TestResult("steiger", float(t), float(p),
                      {"r_jh": r_jh, "r_kh": r_kh, "r_jk": r_jk, "n": n,
                       "delta_rho": r_jh - r_kh, "det": float(det)})


def williams(r_jh: float, r_kh: float, r_jk: float, n: int) -> TestResult:
    if n < 4:
        raise ValueError("n must be >= 4")
    det = (1.0 - r_jh ** 2 - r_kh ** 2 - r_jk ** 2
           + 2.0 * r_jh * r_kh * r_jk)
    r_bar = (r_jh + r_kh) / 2.0
    num = (r_jh - r_kh) * np.sqrt((n - 1) * (1.0 + r_jk))
    den = np.sqrt(2.0 * ((n - 1) / (n - 3)) * det
                  + (r_bar ** 2) * ((1.0 - r_jk) ** 3))
    t = num / den
    p = 2.0 * sps.t.sf(abs(t), df=n - 3)
    return TestResult("williams", float(t), float(p),
                      {"r_jh": r_jh, "r_kh": r_kh, "r_jk": r_jk, "n": n})


def paired_permutation_corr(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray,
                            n_rounds: int = 10_000, method: str = "spearman",
                            seed: int = 42) -> TestResult:
    x_a, x_b, y = map(np.asarray, (x_a, x_b, y))
    ok = ~(np.isnan(x_a) | np.isnan(x_b) | np.isnan(y))
    x_a, x_b, y = x_a[ok], x_b[ok], y[ok]
    n = len(y)
    if n < 4:
        raise ValueError(f"too few complete cases: {n}")

    corr = sps.spearmanr if method == "spearman" else sps.pearsonr

    def rho(u, v):
        return corr(u, v)[0]

    observed = rho(x_a, y) - rho(x_b, y)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_rounds):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, x_b, x_a)
        pb = np.where(swap, x_a, x_b)
        if abs(rho(pa, y) - rho(pb, y)) >= abs(observed):
            count += 1
    p = (count + 1) / (n_rounds + 1)
    return TestResult("paired_permutation", float(observed), float(p),
                      {"n": n, "n_rounds": n_rounds, "method": method,
                       "rho_a": float(rho(x_a, y)), "rho_b": float(rho(x_b, y))})


def bca_bootstrap(data: np.ndarray, statistic, n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 42) -> dict:
    data = np.asarray(data)
    n = len(data)
    theta_hat = statistic(data)

    rng = np.random.default_rng(seed)
    boots = np.array([statistic(data[rng.integers(0, n, n)]) for _ in range(n_boot)])
    boots = boots[~np.isnan(boots)]
    if len(boots) < n_boot // 2:
        raise RuntimeError("bootstrap produced too many NaNs to be trustworthy")

    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / len(boots)), 1.0 - 1.0 / len(boots))
    z0 = sps.norm.ppf(prop)

    jack = np.array([statistic(np.delete(data, i, axis=0)) for i in range(n)])
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    zl, zu = sps.norm.ppf(alpha / 2), sps.norm.ppf(1 - alpha / 2)
    def adj(z):
        return sps.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = adj(zl), adj(zu)
    return {
        "estimate": float(theta_hat),
        "ci_low": float(np.quantile(boots, lo)),
        "ci_high": float(np.quantile(boots, hi)),
        "alpha": alpha, "n_boot": int(len(boots)),
        "z0": float(z0), "acceleration": float(a),
    }


def spearman_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10_000,
                     alpha: float = 0.05, seed: int = 42) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    d = np.column_stack([x[ok], y[ok]])
    out = bca_bootstrap(d, lambda m: sps.spearmanr(m[:, 0], m[:, 1])[0],
                        n_boot=n_boot, alpha=alpha, seed=seed)
    out["n"] = int(ok.sum())
    out["rho_squared"] = float(out["estimate"] ** 2)
    return out

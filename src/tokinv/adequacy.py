"""Section 6: how much meaning is actually lost under romanisation?

Two independent instruments.

1. An adequacy oracle (App. Adequacy Oracle by Fragmentation Tercile):
   LaBSE cross-lingual retrieval of the English source from the native and
   from the romanised target, stratified by how much romanisation raised
   token parity. Input: per-segment retrieval outcomes,
   ``data/derived/adequacy/labse_retrieval_segments.parquet``.

2. Distortion at zero information loss (App. Distortion at Zero Information
   Loss): COMET-22 agreement against the measured round-trip character error
   rate, over ISO 15919 plus four synthetic-noise levels. The intercept at
   CER 0 is what romanisation costs when nothing is lost; the gap to native
   agreement is pure distortion.
   Input: ``data/derived/adequacy/iso_dose_segments.parquet``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sps

from . import stats
from .metaeval import acc_star_eq_pooled
from .paths import DATA, INDIC, INDIC_LANGS, TABLES

A = DATA / "derived" / "adequacy"
NOISE = ["02", "05", "10", "20"]
N_BOOT = 10_000
SEED = 42


# ------------------------------------------------- adequacy oracle ----- #
def _paired_diff_bootstrap(a, b):
    n = len(a)
    rng = np.random.default_rng(SEED)
    boots = np.array([a[idx].mean() - b[idx].mean()
                      for idx in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return a.mean() - b.mean(), lo, hi, bool(lo > 0 or hi < 0)


def labse_stratified() -> pd.DataFrame:
    df = pd.read_parquet(A / "labse_retrieval_segments.parquet")
    rows = []
    for lang in INDIC_LANGS:
        sub = df[df.lang == lang].copy()
        sub["tercile"] = pd.qcut(sub.delta_tp, 3, labels=["low", "mid", "high"])
        for tier in ["low", "mid", "high"]:
            t = sub[sub.tercile == tier]
            n = len(t)
            nat, rom = t.native_correct.to_numpy(), t.romanised_correct.to_numpy()
            drop, dlo, dhi, excl = _paired_diff_bootstrap(nat, rom)
            rng = np.random.default_rng(SEED)
            nb = np.array([nat[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
            rng = np.random.default_rng(SEED + 1)
            rb = np.array([rom[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
            (nlo, nhi), (rlo, rhi) = np.quantile(nb, [0.025, 0.975]), np.quantile(rb, [0.025, 0.975])
            rows.append({"lang": lang, "tercile": tier, "n": n, "mean_delta_tp": t.delta_tp.mean(),
                         "native_acc1": nat.mean(), "native_acc1_ci_low": nlo,
                         "native_acc1_ci_high": nhi, "romanised_acc1": rom.mean(),
                         "romanised_acc1_ci_low": rlo, "romanised_acc1_ci_high": rhi,
                         "drop": drop, "drop_ci_low": dlo, "drop_ci_high": dhi,
                         "drop_excludes_zero": excl})
    return pd.DataFrame(rows)


def labse_regression() -> pd.DataFrame:
    """logit(correct ~ condition + tp): the romanisation effect with token
    parity held fixed, per language and pooled with language fixed effects."""
    df = pd.read_parquet(A / "labse_retrieval_segments.parquet")
    long = pd.concat([
        pd.DataFrame({"lang": df.lang, "segment_id": df.segment_id, "condition": 0,
                      "correct": df.native_correct.astype(int), "tp": df.native_tp}),
        pd.DataFrame({"lang": df.lang, "segment_id": df.segment_id, "condition": 1,
                      "correct": df.romanised_correct.astype(int), "tp": df.romanised_tp}),
    ]).sort_index(kind="mergesort").reset_index(drop=True)

    def fit(data, formula, scope):
        m = smf.logit(formula, data=data).fit(disp=0)
        ci = m.conf_int().loc["condition"]
        return {"scope": scope, "n_obs": len(data), "condition_coef": m.params["condition"],
                "condition_se": m.bse["condition"], "condition_p": m.pvalues["condition"],
                "condition_ci_low": ci[0], "condition_ci_high": ci[1],
                "tp_coef": m.params["tp"], "tp_p": m.pvalues["tp"],
                "converged": m.mle_retvals.get("converged", None)}

    rows = [fit(long[long.lang == l], "correct ~ condition + tp", l) for l in INDIC_LANGS]
    rows.append(fit(long, "correct ~ condition + tp + C(lang)", "POOLED_with_lang_FE"))
    return pd.DataFrame(rows)


# ---------------------------------------- distortion at zero loss ----- #
def iso15919_cer() -> pd.DataFrame:
    """Measured round-trip error of ISO 15919, per language."""
    d = pd.read_parquet(A / "iso_dose_segments.parquet")
    rows = []
    for lang, sub in d.groupby("lang"):
        c = sub.cer_target.dropna()
        rows.append({"lang": lang, "n": int(len(c)), "mean": c.mean(), "std": c.std(),
                     "min": c.min(), "p25": c.quantile(.25), "median": c.median(),
                     "p75": c.quantile(.75), "p90": c.quantile(.90), "p95": c.quantile(.95),
                     "p99": c.quantile(.99), "max": c.max(),
                     "frac_exact_zero": float((c == 0).mean()),
                     "frac_below_0.01": float((c < 0.01).mean()),
                     "mean_cer_reference": sub.cer_reference.dropna().mean()})
    return pd.DataFrame(rows).sort_values("mean")


def _spearman_min10(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    return stats.spearman(a[ok], b[ok]) if ok.sum() >= 10 else float("nan")


def _acc_for(df, col):
    dh, dm = [], []
    for _, g in df.groupby("source_en"):
        m, h = g[col].to_numpy(float), g.mqm_score.to_numpy(float)
        ok = ~np.isnan(m) & ~np.isnan(h)
        m, h = m[ok], h[ok]
        if len(m) < 2:
            continue
        i, j = np.triu_indices(len(m), 1)
        dh.append(h[i] - h[j])
        dm.append(m[i] - m[j])
    return acc_star_eq_pooled(np.concatenate(dh), np.concatenate(dm))[0] if dh else float("nan")


CONDITIONS = ([("native", "comet22_native", None), ("iso", "comet22_iso", "cer_target")]
              + [(f"iso_noise{lv}", f"comet22_iso_noise{lv}", f"cer_noise{lv}") for lv in NOISE]
              + [("indicxlit", "comet22_indicxlit", "cer_indicxlit")])


def dose_response_cer() -> pd.DataFrame:
    d = pd.read_parquet(A / "iso_dose_segments.parquet")
    d = d[d.mqm_valid]
    rows = []
    for lang, sub in d.groupby("lang"):
        for name, scol, ccol in CONDITIONS:
            rows.append({"lang": lang, "condition": name,
                         "mean_cer": 0.0 if ccol is None else float(sub[ccol].mean()),
                         "spearman_vs_mqm": _spearman_min10(sub[scol], sub.mqm_score),
                         "acc_star_eq": _acc_for(sub, scol),
                         "mean_score": float(sub[scol].mean()), "n": int(len(sub))})
    return pd.DataFrame(rows)


def decomposition(dr: pd.DataFrame) -> pd.DataFrame:
    """Linear fit of agreement on CER over the ISO family, per language."""
    rows = []
    for lang, sub in dr.groupby("lang"):
        fam = sub[sub.condition.str.startswith("iso")]
        nat_a = float(sub[sub.condition == "native"].spearman_vs_mqm.iloc[0])
        b, a0 = np.polyfit(fam.mean_cer.to_numpy(), fam.spearman_vs_mqm.to_numpy(), 1)
        ix = sub[sub.condition == "indicxlit"]
        ix_c, ix_a = float(ix.mean_cer.iloc[0]), float(ix.spearman_vs_mqm.iloc[0])
        rows.append({"lang": lang, "A_native": nat_a, "A_intercept_cer0": a0,
                     "distortion_A_nat_minus_A0": nat_a - a0, "slope_per_unit_cer": b,
                     "iso_measured_cer": float(sub[sub.condition == "iso"].mean_cer.iloc[0]),
                     "indicxlit_cer": ix_c, "indicxlit_observed": ix_a,
                     "indicxlit_predicted": a0 + b * ix_c,
                     "indicxlit_residual": ix_a - (a0 + b * ix_c)})
    return pd.DataFrame(rows)


# ---------------------------------------------- Appendix, token parity ---- #
def iso_vs_indicxlit_parity(dr: pd.DataFrame) -> pd.DataFrame:
    """Does near-lossless ISO 15919 fragment XLM-R more than lossy IndicXlit?
    Token parity TP = |t(target)| / |t(source_en)|, paired within segment."""
    from . import tokenisers as T
    count = lambda s: T.n_tokens("xlmr", str(s))
    seg = pd.read_parquet(INDIC / "segments.parquet")
    seg["condition"] = seg["condition"].astype(str)
    nat = seg[seg.condition == "native"].set_index("segment_id")
    rom = seg[seg.condition == "romanised"].set_index("segment_id")
    iso = pd.read_parquet(A / "iso_dose_segments.parquet").set_index("segment_id")
    ids = sorted(set(nat.index) & set(rom.index) & set(iso.index))
    rows = []
    for i in ids:
        en = count(nat.loc[i, "source_en"])
        if en == 0:
            continue
        k = (count(nat.loc[i, "target"]), count(rom.loc[i, "target"]), count(iso.loc[i, "target_iso"]))
        rows.append({"segment_id": i, "lang": str(nat.loc[i, "lang"]),
                     "mqm_valid": bool(nat.loc[i, "mqm_valid"]), "n_tok_en": en,
                     "n_tok_native": k[0], "n_tok_indicxlit": k[1], "n_tok_iso": k[2],
                     "tp_native": k[0] / en, "tp_indicxlit": k[1] / en, "tp_iso": k[2] / en})
    df = pd.DataFrame(rows)
    rho = dr.set_index(["lang", "condition"]).spearman_vs_mqm
    out = []
    for lang in INDIC_LANGS:
        g = df[df.lang == lang]
        d_tp, d_tok = g.tp_iso - g.tp_indicxlit, g.n_tok_iso - g.n_tok_indicxlit
        r_iso, r_ixl = round(rho[(lang, "iso")], 3), round(rho[(lang, "indicxlit")], 3)
        out.append({"lang": lang, "n": len(g), "tp_native": g.tp_native.mean(),
                    "tp_indicxlit": g.tp_indicxlit.mean(), "tp_iso": g.tp_iso.mean(),
                    "tok_native": g.n_tok_native.mean(), "tok_indicxlit": g.n_tok_indicxlit.mean(),
                    "tok_iso": g.n_tok_iso.mean(), "tok_en": g.n_tok_en.mean(),
                    "delta_tp_iso_minus_indicxlit": d_tp.mean(),
                    "delta_tok_iso_minus_indicxlit": d_tok.mean(),
                    "frac_segments_iso_longer": float((d_tok > 0).mean()),
                    "wilcoxon_p_tp": float(sps.wilcoxon(g.tp_iso, g.tp_indicxlit).pvalue),
                    "rho_iso": r_iso, "rho_indicxlit": r_ixl,
                    "iso_agrees_worse": bool(r_iso < r_ixl),
                    "iso_fragments_more": bool(d_tp.mean() > 0)})
    return pd.DataFrame(out), df


def run() -> None:
    labse_stratified().to_csv(TABLES / "labse_stratified_retrieval.csv", index=False)
    labse_regression().to_csv(TABLES / "labse_regression_partial_effect.csv", index=False)
    iso15919_cer().to_csv(TABLES / "iso15919_cer.csv", index=False)
    dr = dose_response_cer()
    dr.to_csv(TABLES / "dose_response_cer.csv", index=False)
    decomposition(dr).to_csv(TABLES / "dose_response_decomposition.csv", index=False)
    summ, segs = iso_vs_indicxlit_parity(dr)
    summ.to_csv(TABLES / "iso_vs_indicxlit_parity.csv", index=False)
    segs.to_csv(TABLES / "iso_vs_indicxlit_parity_segments.csv", index=False)


if __name__ == "__main__":
    run()

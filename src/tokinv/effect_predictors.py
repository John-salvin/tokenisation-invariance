"""Limitations: does any tokeniser property predict the size of the
re-segmentation effect across the 20 Table 1 settings? The predictors,
outcomes and statistic were fixed before computing.

Predictors come from data/derived/effect_predictors/segment_covariates.parquet
(experiments/tokenisation/effect_covariates.py); outcomes and canonical agreement from
the Table 1 data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import resegmentation, stats
from .metaeval import load_panel
from .paths import DATA, TABLES

COV = DATA / "derived" / "effect_predictors" / "segment_covariates.parquet"
PREDICTORS = ["fertility", "token_parity", "canonical_agreement", "headroom"]
OUTCOMES = ["agreement_drop", "monotonicity_rho"]
N_BOOT, N_PERM, SEED = 2000, 10_000, 0


def settings() -> pd.DataFrame:
    """One row per Table 1 setting: the four predictors and the two outcomes."""
    cov = pd.read_parquet(COV)
    panel = load_panel()
    rows = []
    for corpus, labels, setting, file, subset in resegmentation.SETTINGS:
        if corpus == "WMT25 CJK":
            continue
        d = resegmentation.load_setting(file, subset)
        c = cov[cov.setting == setting]
        curve = stats.dose_curve(d)
        if file == "indic_comet22":
            # no canonical rows in re-segmentation: COMET-22's native agreement on the same segments
            p = panel[(panel.lang == subset) & (panel.condition == "native")
                      & panel.segment_id.isin(d.unit.unique())]
            canon = stats.spearman(p.comet_22, p.mqm_score)
        else:
            k = d[d.bucket == "canonical"]
            canon = stats.spearman(k.score, k.human)
        rows.append({"corpus": corpus, "setting": setting, "n_segments": len(c),
                     "fertility": float((c.n_tgt_tok / c.n_tgt_words.clip(lower=1)).mean()),
                     "token_parity": float((c.n_tgt_tok / c.n_src_tok.clip(lower=1)).mean()),
                     "canonical_agreement": canon,
                     "headroom": float(c.headroom.median()),
                     "agreement_drop": float(curve.rho.iloc[0] - curve.rho.iloc[-1]),
                     "monotonicity_rho": stats.monotonicity(d)})
    return pd.DataFrame(rows)


def _spearman(x, y) -> float:
    return stats.spearman(x, y)


def correlations(s: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho across settings, bootstrap interval over settings and a
    two-sided permutation p, for every predictor x outcome."""
    rows = []
    for out in OUTCOMES:
        for pred in PREDICTORS:
            x, y = s[pred].to_numpy(float), s[out].to_numpy(float)
            rho = _spearman(x, y)
            rng = np.random.default_rng(SEED)
            n = len(x)
            boot = []
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                r = _spearman(x[i], y[i])
                if not np.isnan(r):
                    boot.append(r)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rng = np.random.default_rng(SEED)
            perm = np.array([_spearman(x, rng.permutation(y)) for _ in range(N_PERM)])
            p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
            rows.append({"outcome": out, "predictor": pred, "n_settings": n, "rho": rho,
                         "ci_lo": float(lo), "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
                         "perm_p": p})
    return pd.DataFrame(rows)


def within_wmt(s: pd.DataFrame) -> pd.DataFrame:
    """EXPLORATORY, added after seeing correlations(): the same statistic on
    the 13 WMT ESA settings only, to separate language from corpus (the five
    IndicMT Eval settings differ in both)."""
    sub = s[s.corpus.isin(["WMT24", "WMT25"]) & ~s.setting.isin(["SPA", "DEU"])].reset_index(drop=True)
    return correlations(sub).assign(subset="WMT ESA settings (exploratory)")


# script of each setting's target, for the vocabulary appendix
SCRIPT = {"GUJ": "Gujarati", "TAM": "Tamil", "MAL": "Malayalam", "MAR": "Devanagari", "HIN": "Devanagari",
          "SPA": "Latin", "DEU": "Latin", "en-cs": "Latin", "en-uk": "Cyrillic", "en-ru": "Cyrillic",
          "en-is": "Latin", "en-hi": "Devanagari", "en-cs_CZ": "Latin", "en-is_IS": "Latin",
          "en-bho_IN": "Devanagari", "en-uk_UA": "Cyrillic", "en-ar_EG": "Arabic", "en-ru_RU": "Cyrillic",
          "en-et_EE": "Latin", "en-it_IT": "Latin", "en-zh_CN": "Han", "en-ja_JP": "Kana", "en-ko_KR": "Hangul"}


def _malayalam_mean_token_len() -> float:
    """The vocabulary table omits Malayalam; same rule, Malayalam block
    (the same rule as the vocabulary table)."""
    from .tokenisers import get
    from .tokeniser_stats import script_of
    tot = lens = 0
    for piece in get("xlmr").get_vocab():
        s = piece.lstrip("\u2581")
        sc = {("Malayalam" if 0x0D00 <= ord(c) <= 0x0D7F else script_of(c)) for c in s} - {None}
        if s and sc == {"Malayalam"}:
            tot += 1
            lens += len(s)
    return lens / tot


def vocab_ceiling() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vocabulary appendix: median fragmentation ceiling per setting (Table 1
    settings and the three CJK settings, on the same canonical rows as their
    re-segmentation runs), per script, and its correlation with mean vocabulary token
    length across scripts."""
    from scipy import stats as sps
    from .tokeniser_stats import vocab_by_script
    cov = pd.read_parquet(COV)
    rows = []
    for setting, g in cov.groupby("setting", sort=False):
        h = g.headroom
        rows.append({"setting": setting, "script": SCRIPT[setting], "n_rows": len(g),
                     "n_unsegmentable": int(h.isna().sum()), "n_scored": int(h.notna().sum()),
                     "median_ceiling": float(h.median())})
    st = pd.DataFrame(rows)
    vt = vocab_by_script().set_index("script").mean_token_len.to_dict()
    vt["Malayalam"] = round(_malayalam_mean_token_len(), 2)
    per = st.groupby("script").median_ceiling.median()
    x = [vt[k] for k in per.index]
    cjk = st.setting.isin(["en-zh_CN", "en-ja_JP", "en-ko_KR"])
    summ = pd.DataFrame([{
        "n_scripts": len(per), "pearson": float(sps.pearsonr(x, per.values)[0]),
        "pearson_p": float(sps.pearsonr(x, per.values)[1]),
        "table1_min": float(st[~cjk].median_ceiling.min()), "table1_max": float(st[~cjk].median_ceiling.max()),
        "cjk_min": float(st[cjk].median_ceiling.min()), "cjk_max": float(st[cjk].median_ceiling.max())}])
    return st, summ


def run() -> None:
    s = settings()
    s.to_csv(TABLES / "effect_predictors_settings.csv", index=False)
    c = correlations(s)
    c.to_csv(TABLES / "effect_predictors_correlations.csv", index=False)
    within_wmt(s).to_csv(TABLES / "effect_predictors_within_wmt.csv", index=False)
    vs, vsum = vocab_ceiling()
    vs.to_csv(TABLES / "vocab_ceiling_settings.csv", index=False)
    vsum.to_csv(TABLES / "vocab_ceiling_summary.csv", index=False)
    summary = pd.DataFrame([{"max_abs_rho": float(c.rho.abs().max()),
                             "n_excluding_zero": int(c.excludes_zero.sum()),
                             "min_perm_p": float(c.perm_p.min())}])
    summary.to_csv(TABLES / "effect_predictors_summary.csv", index=False)


if __name__ == "__main__":
    run()

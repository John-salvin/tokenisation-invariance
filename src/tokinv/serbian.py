"""Section 5.2 and App. Serbian: a change of script without fragmentation.

WMT25 English-Serbian, scored by COMET-22 in its native Cyrillic and in a
lossless Latin (Gaj) transliteration. The Latin form uses slightly *fewer*
XLM-R tokens, and agreement does not change: the control for the
fragmentation account of the romanisation drop.

Input: ``data/derived/serbian/*.parquet``, one row per scored (system,
segment) pair with the ESA score, both COMET-22 scores, both XLM-R token
counts and the round-trip check. Produced by
experiments/script/serbian.py.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import offtarget, stats
from .metaeval import acc_star_eq_pooled
from .paths import DATA, TABLES

SCORES = DATA / "derived" / "serbian" / "en-sr_Cyrl_RS_comet22_cyrillic_latin.parquet"


def subsets(df: pd.DataFrame, on_target: bool = True):
    # "fits_512": both forms fit XLM-R's 512 positions, so nothing is truncated.
    # on_target: drop systems that wrote Serbian in Latin script or another
    # language (tokinv.offtarget); their "Cyrillic" condition is not Cyrillic.
    # the human reference is scored as a "system" in the score file; COMET-22
    # would compare it with itself, so it is never a system here
    df = df[~df.system.str.lower().str.startswith("ref")]
    if on_target:
        df = df[~df.system.isin(offtarget.off_target("en-sr_Cyrl_RS"))]
    return [("all", df), ("fits_512", df[~df.over_limit])]


def agreement(df: pd.DataFrame, on_target: bool = True) -> pd.DataFrame:
    rows = []
    for label, d in subsets(df, on_target):
        for cond in ("cyrillic", "latin"):
            col = f"comet22_{cond}"
            dh, dm = [], []
            for _, g in d.groupby("seg_idx"):
                m, h = g[col].to_numpy(float), g.esa.to_numpy(float)
                if len(m) < 2:
                    continue
                i, j = np.triu_indices(len(m), 1)
                dh.append(h[i] - h[j])
                dm.append(m[i] - m[j])
            acc, eps = acc_star_eq_pooled(np.concatenate(dh), np.concatenate(dm))
            rows.append({"subset": label, "condition": cond, "n": len(d),
                         "spearman_vs_esa": stats.spearman(d[col], d.esa),
                         "acc_star_eq": acc, "epsilon": eps,
                         "n_pairs": int(sum(len(x) for x in dh))})
    return pd.DataFrame(rows)


def ranking_flips(df: pd.DataFrame, on_target: bool = True) -> pd.DataFrame:
    rows = []
    for label, d in subsets(df, on_target):
        mc = d.groupby("system").comet22_cyrillic.mean()
        ml = d.groupby("system").comet22_latin.mean()
        me = d.groupby("system").esa.mean()
        common = sorted(set(mc.index) & set(ml.index))
        pairs = list(itertools.combinations(common, 2))
        flips = sum(np.sign(mc[a] - mc[b]) != np.sign(ml[a] - ml[b]) for a, b in pairs)
        rows.append({"subset": label, "n_systems": len(common), "n_pairs": len(pairs),
                     "n_flips": int(flips),
                     "spearman_syslevel_cyr_vs_esa": stats.spearman(mc[common], me[common]),
                     "spearman_syslevel_lat_vs_esa": stats.spearman(ml[common], me[common])})
    return pd.DataFrame(rows)


def summary() -> pd.DataFrame:
    """The Serbian numbers the paper prints, for both system sets: systems,
    rows, mean token counts, the share of outputs with any Latin character,
    and the Latin-minus-Cyrillic agreement difference with its segment
    bootstrap interval."""
    df = pd.read_parquet(SCORES)
    rows = []
    for label, on in [("on_target", True), ("all_systems", False)]:
        d = dict(subsets(df, on))["all"]
        f = lambda y: stats.spearman(y.comet22_latin, y.esa) - stats.spearman(y.comet22_cyrillic, y.esa)
        lo, hi = stats.segment_bootstrap(d, f, "seg_idx")
        fl = ranking_flips(df, on).iloc[0]
        rows.append({"systems": label, "n_systems": d.system.nunique(), "n_rows": len(d),
                     "cyrillic": stats.spearman(d.comet22_cyrillic, d.esa),
                     "latin": stats.spearman(d.comet22_latin, d.esa), "diff": f(d),
                     "diff_lo": lo, "diff_hi": hi, "n_flips": int(fl.n_flips), "n_pairs": int(fl.n_pairs),
                     "ntok_cyr": d.ntok_cyr.mean(), "ntok_lat": d.ntok_lat.mean(),
                     "tok_change_pct": 100 * (d.ntok_lat.mean() / d.ntok_cyr.mean() - 1),
                     "latin_char_pct": 100 * float((d.cyr_frac < 1).mean()),
                     # round trip Latin -> Cyrillic on purely Cyrillic outputs
                     "roundtrip_pure_cyr_pct": 100 * float(d[d.cyr_frac == 1].roundtrip_exact.mean())})
    return pd.DataFrame(rows)


def run() -> None:
    df = pd.read_parquet(SCORES)
    df.to_csv(TABLES / "serbian_scores.csv", index=False)
    agreement(df).to_csv(TABLES / "serbian_agreement.csv", index=False)
    ranking_flips(df).to_csv(TABLES / "serbian_flips.csv", index=False)
    agreement(df, False).to_csv(TABLES / "serbian_agreement_all_systems.csv", index=False)
    ranking_flips(df, False).to_csv(TABLES / "serbian_flips_all_systems.csv", index=False)
    summary().to_csv(TABLES / "serbian_summary.csv", index=False)


if __name__ == "__main__":
    run()

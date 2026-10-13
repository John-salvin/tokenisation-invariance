"""Section 4.2 and App. Near-Canonical Token Counts.

Re-segmentations whose token count stays within 5 percent of canonical
(ratio in [1.00, 1.05]). If agreement still falls with almost no extra tokens,
the fault is in *which* tokens, not how many. Each segment's five seeded
re-segmentations are averaged first, so every segment counts once: the earlier
analysis treated the five as independent and overstated n five-fold.

Input: data/derived/equal_count/equal_count_comet22.parquet (COMET-22 on each
near-canonical re-segmentation; experiments/tokenisation/run_equal_count_sample.py
and run_equal_count_score.py).
"""
from __future__ import annotations

import pandas as pd

from . import corrtests as CT
from .paths import DATA, INDIC, INDIC_LANGS, TABLES

BUCKET = "[1.00,1.05]"


def significance() -> pd.DataFrame:
    seg = pd.read_parquet(INDIC / "segments.parquet")
    seg = seg[seg.mqm_valid]
    nat = seg[seg.condition == "native"].set_index(["lang", "segment_id"])
    eq = pd.read_parquet(DATA / "derived" / "equal_count" / "equal_count_comet22.parquet")
    eq = eq[eq.bucket == BUCKET]
    rows = []
    for lang in INDIC_LANGS:
        b = (eq[eq.lang == lang].groupby("segment_id")
               .agg(bucket_comet22=("comet22", "mean"), n_seeds=("comet22", "size")).reset_index())
        b = b[b.segment_id.isin(nat.loc[lang].index)]
        # the workbook stores COMET-22 x 100; the re-scored samples are on [0, 1]
        canon = nat.loc[lang].loc[b.segment_id, "comet_22"].to_numpy() / 100.0
        mqm = nat.loc[lang].loc[b.segment_id, "mqm_score"].to_numpy()
        buck = b.bucket_comet22.to_numpy()
        n = len(b)
        r_jh = float(pd.Series(canon).corr(pd.Series(mqm), method="spearman"))
        r_kh = float(pd.Series(buck).corr(pd.Series(mqm), method="spearman"))
        r_jk = float(pd.Series(canon).corr(pd.Series(buck), method="spearman"))
        mrr = CT.meng_rosenthal_rubin(r_jh, r_kh, r_jk, n)
        st = CT.steiger(r_jh, r_kh, r_jk, n)
        perm = CT.paired_permutation_corr(canon, buck, mqm, n_rounds=10_000)
        verdict = lambda p: "distinguishable" if p < 0.05 else "not distinguishable"
        rows.append({"lang": lang, "n_paired_segments": n,
                     "mean_seeds_per_segment": float(b.n_seeds.mean()),
                     "r_jh_canonical_rho": r_jh, "r_kh_bucket_rho": r_kh,
                     "r_jk_canon_bucket_corr": r_jk, "mrr_z": mrr.statistic, "mrr_p": mrr.p_value,
                     "steiger_t": st.statistic, "steiger_p": st.p_value, "perm_p": perm.p_value,
                     "new_verdict_mrr_alpha05": verdict(mrr.p_value),
                     "new_verdict_steiger_alpha05": verdict(st.p_value),
                     "new_verdict_perm_alpha05": verdict(perm.p_value)})
    return pd.DataFrame(rows)


def run() -> None:
    significance().to_csv(TABLES / "equal_count_significance_corrected.csv", index=False)


if __name__ == "__main__":
    run()

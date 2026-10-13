#!/usr/bin/env python3
"""MetricX-24: check scoring from token ids against text, then score both conditions.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import metrics_panel as MP
import provenance as PROV
import tokenisers as T

N_VERIFY = 200
TOL = 1e-6
TOK = "mt5"


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)

    print("=" * 78); print("Step 1: equivalence check, 200 segments"); print("=" * 78)
    nat_valid = seg[(seg.condition == "native") & seg.mqm_valid]
    sample = nat_valid.sample(n=min(N_VERIFY, len(nat_valid)), random_state=C.SEED)
    rows = []
    for _, r in sample.iterrows():
        src, mt, ref = str(r.source_en), str(r.target), str(r.reference)
        t = MP.metricx_score_from_text(src, mt, ref)
        src_ids = T.tokenise(src, TOK, add_special_tokens=False)
        mt_ids = T.tokenise(mt, TOK, add_special_tokens=False)
        ref_ids = T.tokenise(ref, TOK, add_special_tokens=False)
        i = MP.metricx_score_from_ids(src_ids, mt_ids, ref_ids)
        rows.append({"segment_id": r.segment_id, "lang": r.lang,
                     "score_from_text": t, "score_from_ids": i, "abs_diff": abs(t - i)})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(C.OUT_TABLES / "metricx24_equivalence_check.csv", index=False)
    max_diff = vdf.abs_diff.max()
    n_fail = int((vdf.abs_diff > TOL).sum())
    print(f"checked {len(vdf)} segments; max |diff| = {max_diff:.2e}; n exceeding {TOL:.0e}: {n_fail}")
    if n_fail:
        raise MP.MetricsPanelError(
            f"STOP: MetricX-24 score_from_ids does not match score_from_text to {TOL:.0e} "
            f"on {n_fail}/{len(vdf)} segments (max diff {max_diff:.2e}). Not proceeding.")
    print(f"PASSED: exact/near-exact match on all {len(vdf)} segments.")

    print(); print("=" * 78); print("Step 2: panel scoring, native + romanised, all languages"); print("=" * 78)
    valid = seg[seg.mqm_valid].copy()
    scores = []
    for idx, r in valid.iterrows():
        src, mt, ref = str(r.source_en), str(r.target), str(r.reference)
        s = MP.metricx_score_from_text(src, mt, ref)
        scores.append(s)
    valid["metricx24"] = scores
    valid[["segment_id", "lang", "condition", "metricx24", "mqm_score"]].to_csv(
        C.OUT_TABLES / "metricx24_scores.csv", index=False)

    print(f"{'lang':6s} {'cond':10s} {'n':>5s} {'mean_metricx24':>15s} {'spearman_vs_mqm':>16s}")
    from scipy import stats as sps
    summary_rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = valid[(valid.lang == lang) & (valid.condition == cond)]
            rho = sps.spearmanr(sub.metricx24, sub.mqm_score)[0]
            print(f"{lang:6s} {cond:10s} {len(sub):5d} {sub.metricx24.mean():15.4f} {rho:16.4f}")
            summary_rows.append({"lang": lang, "condition": cond, "n": len(sub),
                                 "mean_metricx24": float(sub.metricx24.mean()),
                                 "spearman_vs_mqm": float(rho)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(C.OUT_TABLES / "metricx24_summary.csv", index=False)

    PROV.register(
        "metricx24_panel", {
            "equivalence": {"n_checked": len(vdf), "max_abs_diff": float(max_diff),
                            "tolerance": TOL, "passed": n_fail == 0},
            "summary": summary_df.to_dict("records"),
        },
        PROV.stamp(Path(__file__), modules=["metrics_panel", "tokenisers", "constants"],
                   seed=C.SEED,
                   note=("MetricX-24 (only non-XLM-R vocab in the panel), equivalence-verified "
                         "then scored on all valid native+romanised segments; architecture and "
                         "input template reproduced from github.com/google-research/metricx "
                         "primary source, read directly, not summarised")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

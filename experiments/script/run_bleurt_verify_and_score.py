#!/usr/bin/env python3
"""BLEURT-20: check scoring from token ids against text, then score both conditions.

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

N_VERIFY = 200
TOL = 1e-6


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    model, tokenizer = MP.load_bleurt20()

    print("=" * 78); print("Step 1: equivalence check, 200 segments"); print("=" * 78)
    nat_valid = seg[(seg.condition == "native") & seg.mqm_valid]
    sample = nat_valid.sample(n=min(N_VERIFY, len(nat_valid)), random_state=C.SEED)
    rows = []
    for _, r in sample.iterrows():
        mt, ref = str(r.target), str(r.reference)
        t = MP.bleurt_score_from_text(ref, mt)
        ref_ids = tokenizer(ref, add_special_tokens=False)["input_ids"]
        mt_ids = tokenizer(mt, add_special_tokens=False)["input_ids"]
        i = MP.bleurt_score_from_ids(ref_ids, mt_ids)
        rows.append({"segment_id": r.segment_id, "lang": r.lang,
                     "score_from_text": t, "score_from_ids": i, "abs_diff": abs(t - i)})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(C.OUT_TABLES / "bleurt20_equivalence_check.csv", index=False)
    max_diff = vdf.abs_diff.max()
    n_fail = int((vdf.abs_diff > TOL).sum())
    print(f"checked {len(vdf)} segments; max |diff| = {max_diff:.2e}; n exceeding {TOL:.0e}: {n_fail}")
    if n_fail:
        raise MP.MetricsPanelError(
            f"STOP: BLEURT-20 score_from_ids does not match score_from_text to {TOL:.0e} "
            f"on {n_fail}/{len(vdf)} segments (max diff {max_diff:.2e}). Not proceeding.")
    print(f"PASSED: exact/near-exact match on all {len(vdf)} segments.")

    print(); print("=" * 78); print("Step 2: panel scoring, native + romanised, all languages"); print("=" * 78)
    valid = seg[seg.mqm_valid].copy()
    scores = []
    for _, r in valid.iterrows():
        mt, ref = str(r.target), str(r.reference)
        scores.append(MP.bleurt_score_from_text(ref, mt))
    valid["bleurt20"] = scores
    valid[["segment_id", "lang", "condition", "bleurt20", "mqm_score"]].to_csv(
        C.OUT_TABLES / "bleurt20_scores.csv", index=False)

    from scipy import stats as sps
    print(f"{'lang':6s} {'cond':10s} {'n':>5s} {'mean_bleurt20':>14s} {'spearman_vs_mqm':>16s}")
    summary_rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = valid[(valid.lang == lang) & (valid.condition == cond)]
            rho = sps.spearmanr(sub.bleurt20, sub.mqm_score)[0]
            print(f"{lang:6s} {cond:10s} {len(sub):5d} {sub.bleurt20.mean():14.4f} {rho:16.4f}")
            summary_rows.append({"lang": lang, "condition": cond, "n": len(sub),
                                 "mean_bleurt20": float(sub.bleurt20.mean()),
                                 "spearman_vs_mqm": float(rho)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(C.OUT_TABLES / "bleurt20_summary.csv", index=False)

    PROV.register(
        "bleurt20_panel", {
            "equivalence": {"n_checked": len(vdf), "max_abs_diff": float(max_diff),
                            "tolerance": TOL, "passed": n_fail == 0},
            "summary": summary_df.to_dict("records"),
        },
        PROV.stamp(Path(__file__), modules=["metrics_panel", "constants"], seed=C.SEED,
                   note=("BLEURT-20, equivalence-verified then scored on all valid "
                         "native+romanised segments; bleurt_pytorch package staged from "
                         "github.com/lucadiliello/bleurt-pytorch")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

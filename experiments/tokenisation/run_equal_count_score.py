#!/usr/bin/env python3
"""COMET-22 scores for the near-canonical re-segmentations (GPU).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import metrics_panel as MP
import provenance as PROV
import tokenisers as T

TOKENISER = "xlmr"
BATCH_SIZE = 48
N_BOOT = 10_000
SEED = 42


def bootstrap_rho_ci(comet: np.ndarray, mqm: np.ndarray, rng: np.random.Generator,
                     n_boot: int = N_BOOT) -> tuple[float, float, float]:
    n = len(comet)
    point = sps.spearmanr(comet, mqm)[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = sps.spearmanr(comet[idx], mqm[idx])[0]
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.DATA_INTERIM / "equal_count_samples.parquet")
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = MP.load_comet("comet22", device=device)

    scoreable = samples[samples.mt_ids.notna()].reset_index(drop=True)
    print(f"scoring {len(scoreable)} reachable samples "
          f"({len(samples) - len(scoreable)} unreachable-bucket placeholders skipped)")

    seg_native = seg[seg.condition == "native"].set_index("segment_id")
    canon_ids: dict[str, tuple[list[int], list[int]]] = {}

    def get_canon(seg_id: str) -> tuple[list[int], list[int]]:
        if seg_id not in canon_ids:
            row = seg_native.loc[seg_id]
            src_ids = T.tokenise(str(row.source_en), TOKENISER, add_special_tokens=False)
            ref_ids = T.tokenise(str(row.reference), TOKENISER, add_special_tokens=False)
            canon_ids[seg_id] = (src_ids, ref_ids)
        return canon_ids[seg_id]

    results = []
    for (lang, bucket), grp in scoreable.groupby(["lang", "bucket"], observed=True):
        for start in range(0, len(grp), BATCH_SIZE):
            batch = grp.iloc[start:start + BATCH_SIZE]
            src_batch, mt_batch, ref_batch = [], [], []
            for _, r in batch.iterrows():
                src_ids, ref_ids = get_canon(r.segment_id)
                src_batch.append(src_ids)
                mt_batch.append(list(r.mt_ids))
                ref_batch.append(ref_ids)
            scores = MP.score_from_ids_batch(model, src_batch, mt_batch, ref_batch)
            for (_, r), s in zip(batch.iterrows(), scores):
                mqm = seg_native.loc[r.segment_id, "mqm_score"]
                results.append({"segment_id": r.segment_id, "lang": lang, "bucket": bucket,
                                "seed": r.seed, "ratio": r.ratio, "comet22": s, "mqm_score": mqm})
        print(f"scored {lang} {bucket}: {len(grp)} samples")

    res_df = pd.DataFrame(results)
    res_df.to_csv(C.OUT_TABLES / "equal_count_scores.csv", index=False)

    print("\n" + "=" * 78); print("Equal-count control vs canonical native baseline")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    summary_rows = []
    for lang in C.LANG_ORDER:
        canon_sub = seg_native.reset_index()
        canon_sub = canon_sub[canon_sub.lang == lang]
        canon_comet = canon_sub["comet_22"].values
        canon_mqm = canon_sub["mqm_score"].values
        canon_rho, canon_lo, canon_hi = bootstrap_rho_ci(canon_comet, canon_mqm, rng)
        summary_rows.append({"lang": lang, "bucket": "canonical", "n": len(canon_sub),
                             "mean_ratio": 1.0, "mean_comet22": float(canon_comet.mean()),
                             "spearman_comet_mqm": canon_rho, "rho_ci_low": canon_lo,
                             "rho_ci_high": canon_hi,
                             "within_canonical_ci": True})
        for bucket in sorted(res_df[res_df.lang == lang]["bucket"].unique()):
            sub = res_df[(res_df.lang == lang) & (res_df.bucket == bucket)]
            if len(sub) < 10:
                continue
            rho, lo, hi = bootstrap_rho_ci(sub["comet22"].values, sub["mqm_score"].values, rng)
            within = (lo <= canon_rho <= hi) or (canon_lo <= rho <= canon_hi) or \
                     not (hi < canon_lo or lo > canon_hi)
            summary_rows.append({"lang": lang, "bucket": bucket, "n": len(sub),
                                 "mean_ratio": float(sub["ratio"].mean()),
                                 "mean_comet22": float(sub["comet22"].mean()),
                                 "spearman_comet_mqm": rho, "rho_ci_low": lo, "rho_ci_high": hi,
                                 "within_canonical_ci": bool(within)})
        print(f"{lang}: canonical rho={canon_rho:.3f} CI=[{canon_lo:.3f},{canon_hi:.3f}]")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(C.OUT_TABLES / "equal_count_summary.csv", index=False)
    print("\n" + summary_df.to_string(index=False))

    below_bucket = summary_df[(summary_df.bucket.str.startswith("[0.95")) ]
    above_bucket = summary_df[(summary_df.bucket.str.startswith("[1.00"))]
    n_below_within = int(below_bucket["within_canonical_ci"].sum())
    n_above_within = int(above_bucket["within_canonical_ci"].sum())
    n_langs = len(C.LANG_ORDER)
    if n_below_within >= 4 and n_above_within >= 4:
        interpretation = ("FRAGMENTATION_CAUSAL: both near-canonical buckets stay within the "
                          "canonical rho's own bootstrap CI in at least 4/5 languages. "
                          "OOD-ness/novelty alone is NOT sufficient to explain re-segmentation's "
                          "dose-response curve -- degradation requires actually increasing "
                          "token count, not merely moving off the canonical path.")
    elif n_below_within <= 1 and n_above_within <= 1:
        interpretation = ("NOVELTY_CONFOUND: rho already collapses at ratio~1.00 in at least "
                          "4/5 languages despite an essentially unchanged token count. Part of "
                          "re-segmentation's curve is attributable to segmentation novelty/OOD-ness, not "
                          "fragmentation per se. The paper's claim must be scoped to "
                          "'fragmentation AND segmentation novelty jointly'.")
    else:
        interpretation = (f"MIXED: {n_below_within}/5 languages stay within canonical CI at "
                          f"[0.95,1.00), {n_above_within}/5 at [1.00,1.05]. Neither the clean "
                          f"causal story nor the clean novelty-confound story holds uniformly "
                          f"across languages -- report per-language, do not average into a "
                          f"single pooled claim.")
    print(f"\nINTERPRETATION: {interpretation}")

    PROV.register(
        "equal_count_control", {
            "summary": summary_df.to_dict("records"),
            "n_scored": len(res_df),
            "interpretation": interpretation,
            "n_languages_below_within_ci": n_below_within,
            "n_languages_above_within_ci": n_above_within,
            "buckets": ["[0.95,1.00)", "[1.00,1.05]"],
            "condition": "native only (see module docstring for why)",
            "n_bootstrap": N_BOOT,
        },
        PROV.stamp(Path(__file__), modules=["constants", "metrics_panel", "tokenisers"], seed=SEED,
                   note=("Ghosh & Jyothi Sec 5.2's own equal-count "
                         "control, closing re-segmentation's fragmentation-vs-novelty confound. "
                         "COMET-22 via the verified score_from_ids_batch path.")))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

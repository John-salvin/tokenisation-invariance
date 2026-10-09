#!/usr/bin/env python3
"""COMET-22 scores for the IndicMT Eval re-segmentations (GPU).

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
RHO_CROSSING = 0.40


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid].copy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model = MP.load_comet("comet22", device=device)

    scoreable = samples[samples.mt_ids.notna()].reset_index(drop=True)
    print(f"scoring {len(scoreable)} reachable samples "
          f"({len(samples) - len(scoreable)} unreachable-bucket placeholders skipped)")

    canon_ids: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    seg_by_id_cond = seg.set_index(["segment_id", "condition"])

    def get_canon_cond(seg_id: str, direction: str) -> tuple[list[int], list[int]]:
        cond = "native" if direction == "forward" else "romanised"
        key = (seg_id, cond, direction)
        if key not in canon_ids:
            row = seg_by_id_cond.loc[(seg_id, cond)]
            src_ids = T.tokenise(str(row.source_en), TOKENISER, add_special_tokens=False)
            ref_ids = T.tokenise(str(row.reference), TOKENISER, add_special_tokens=False)
            canon_ids[key] = (src_ids, ref_ids)
        return canon_ids[key]

    results = []
    groups = list(scoreable.groupby(["lang", "direction", "bucket"], observed=True))
    for (lang, direction, bucket), grp in groups:
        for start in range(0, len(grp), BATCH_SIZE):
            batch = grp.iloc[start:start + BATCH_SIZE]
            src_batch, mt_batch, ref_batch = [], [], []
            for _, r in batch.iterrows():
                src_ids, ref_ids = get_canon_cond(r.segment_id, direction)
                src_batch.append(src_ids)
                mt_batch.append(list(r.mt_ids))
                ref_batch.append(ref_ids)
            scores = MP.score_from_ids_batch(model, src_batch, mt_batch, ref_batch)
            for (_, r), s in zip(batch.iterrows(), scores):
                cond = "native" if direction == "forward" else "romanised"
                mqm = seg_by_id_cond.loc[(r.segment_id, cond), "mqm_score"]
                results.append({"segment_id": r.segment_id, "lang": lang, "direction": direction,
                                "bucket": bucket, "seed": r.seed, "ratio": r.ratio,
                                "comet22": s, "mqm_score": mqm})
        print(f"scored {lang} {direction} {bucket}: {len(grp)} samples")

    res_df = pd.DataFrame(results)
    res_df.to_csv(C.OUT_TABLES / "reseg_scores.csv", index=False)

    reach_rows = []
    for (lang, direction, bucket), grp in samples.groupby(["lang", "direction", "bucket"], observed=True):
        n_total = len(grp)
        n_reachable = grp.mt_ids.notna().sum()
        reach_rows.append({"lang": lang, "direction": direction, "bucket": bucket,
                           "n_total": n_total, "n_reachable": n_reachable,
                           "pct_reachable": 100.0 * n_reachable / n_total if n_total else 0.0})
    reach_df = pd.DataFrame(reach_rows)
    reach_df.to_csv(C.OUT_TABLES / "reseg_bucket_reachability.csv", index=False)
    print("\nBucket reachability:")
    print(reach_df.to_string(index=False))

    print("\n" + "=" * 78); print("Dose-response: mean COMET-22 and Spearman(COMET,MQM) per bucket, per language")
    print("=" * 78)
    curve_rows = []
    for lang in C.LANG_ORDER:
        for direction in ["forward", "reverse"]:
            sub_dir = res_df[(res_df.lang == lang) & (res_df.direction == direction)]
            for bucket in sub_dir.bucket.unique():
                sub = sub_dir[sub_dir.bucket == bucket]
                if len(sub) < 4:
                    continue
                mean_ratio = sub.ratio.mean()
                mean_comet = sub.comet22.mean()
                rho = sps.spearmanr(sub.comet22, sub.mqm_score)[0] if sub.comet22.nunique() > 1 else float("nan")
                curve_rows.append({"lang": lang, "direction": direction, "bucket": bucket,
                                   "n": len(sub), "mean_ratio": mean_ratio,
                                   "mean_comet22": mean_comet, "spearman_comet_mqm": rho})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(C.OUT_TABLES / "reseg_dose_response.csv", index=False)
    print(curve_df.sort_values(["lang", "direction", "mean_ratio"]).to_string(index=False))

    print("\n" + "=" * 78); print(f"rho={RHO_CROSSING} crossing point (linear interp on the forward dose-response curve)")
    print("=" * 78)
    crossing_rows = []
    for lang in C.LANG_ORDER + ["POOLED"]:
        if lang == "POOLED":
            sub = curve_df[curve_df.direction == "forward"].groupby("bucket", observed=True).agg(
                mean_ratio=("mean_ratio", "mean"), spearman_comet_mqm=("spearman_comet_mqm", "mean")
            ).reset_index()
        else:
            sub = curve_df[(curve_df.lang == lang) & (curve_df.direction == "forward")]
        sub = sub.sort_values("mean_ratio").dropna(subset=["spearman_comet_mqm"])
        crossing = None
        for i in range(len(sub) - 1):
            r0, r1 = sub.iloc[i], sub.iloc[i + 1]
            if (r0.spearman_comet_mqm - RHO_CROSSING) * (r1.spearman_comet_mqm - RHO_CROSSING) <= 0 \
               and r0.spearman_comet_mqm != r1.spearman_comet_mqm:
                frac = (RHO_CROSSING - r0.spearman_comet_mqm) / (r1.spearman_comet_mqm - r0.spearman_comet_mqm)
                crossing = r0.mean_ratio + frac * (r1.mean_ratio - r0.mean_ratio)
                break
        crossing_rows.append({"lang": lang, "rho_040_crossing_ratio": crossing,
                              "n_buckets_with_data": len(sub),
                              "rho_range": f"[{sub.spearman_comet_mqm.min():.3f}, {sub.spearman_comet_mqm.max():.3f}]" if len(sub) else "n/a"})
        print(f"{lang:8s} crossing={crossing} n_buckets={len(sub)} "
              f"rho_range=[{sub.spearman_comet_mqm.min() if len(sub) else float('nan'):.3f}, "
              f"{sub.spearman_comet_mqm.max() if len(sub) else float('nan'):.3f}]")
    crossing_df = pd.DataFrame(crossing_rows)
    crossing_df.to_csv(C.OUT_TABLES / "reseg_rho040_crossing.csv", index=False)

    print("\n" + "=" * 78); print("Monotonicity check: does Spearman(COMET,MQM) decrease as ratio increases?")
    print("=" * 78)
    mono_rows = []
    for lang in C.LANG_ORDER:
        sub = curve_df[(curve_df.lang == lang) & (curve_df.direction == "forward")].sort_values("mean_ratio")
        sub = sub.dropna(subset=["spearman_comet_mqm"])
        if len(sub) < 3:
            mono_rows.append({"lang": lang, "spearman_ratio_vs_rho": None, "n": len(sub)})
            continue
        corr = sps.spearmanr(sub.mean_ratio, sub.spearman_comet_mqm)[0]
        mono_rows.append({"lang": lang, "spearman_ratio_vs_rho": corr, "n": len(sub)})
        print(f"{lang}: Spearman(ratio, rho_comet_mqm) across buckets = {corr:.3f} (negative = degrades with fragmentation, as predicted)")
    mono_df = pd.DataFrame(mono_rows)

    PROV.register(
        "reseg_dose_response", {
            "n_scored": len(res_df),
            "bucket_reachability": reach_df.to_dict("records"),
            "dose_response_curve": curve_df.to_dict("records"),
            "rho040_crossing": crossing_df.to_dict("records"),
            "monotonicity_per_lang": mono_df.to_dict("records"),
            "metricx24_not_scored_reason": ("mT5 tokenizer has 256 byte-fallback vocab entries; "
                                            "noncanonical.py's MDD does not model byte-fallback "
                                            "segmentation (verified via build_tries raising on mt5, "
                                            "not assumed); would need a byte-offset MDD redesign, "
                                            "not attempted -- see run_reseg_sample.py docstring"),
        },
        PROV.stamp(Path(__file__), modules=["metrics_panel", "tokenisers", "constants"], seed=C.SEED,
                   note="re-segmentation dose-response, COMET-22 only, XLM-R vocabulary, forward+reverse directions"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

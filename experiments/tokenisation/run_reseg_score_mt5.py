#!/usr/bin/env python3
"""MetricX-24 scores for the mT5 re-segmentations (GPU).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import metrics_panel as MP
import provenance as PROV
import tokenisers as T

TOKENISER = "mt5"


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(C.DATA_INTERIM / "reseg_samples_mt5.parquet")
    seg = pd.read_parquet(Path.home() / "tokinv_work/repo/panel-pipeline/data/interim/segments.parquet")
    seg = seg[seg.mqm_valid].copy()
    seg_by_id_cond = seg.set_index(["segment_id", "condition"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    MP.load_metricx24()

    scoreable = samples[samples.mt_ids.notna()].reset_index(drop=True)
    print(f"scoring {len(scoreable)} reachable samples "
          f"({len(samples) - len(scoreable)} unreachable-bucket placeholders skipped)")

    out_path = C.OUT_TABLES / "reseg_scores_mt5.csv"
    results = []
    if out_path.exists():
        prev = pd.read_csv(out_path)
        results = prev.to_dict("records")
        done_keys = set(zip(prev.segment_id, prev.bucket, prev.seed))
        print(f"resuming: {len(results)} rows already scored")
    else:
        done_keys = set()

    canon_ids: dict[tuple[str, str], tuple[list[int], list[int]]] = {}

    def get_canon(seg_id: str) -> tuple[list[int], list[int]]:
        key = (seg_id, "native")
        if key not in canon_ids:
            row = seg_by_id_cond.loc[(seg_id, "native")]
            src_ids = T.tokenise(str(row.source_en), TOKENISER, add_special_tokens=False)
            ref_ids = T.tokenise(str(row.reference), TOKENISER, add_special_tokens=False)
            canon_ids[key] = (src_ids, ref_ids)
        return canon_ids[key]

    t0 = time.time()
    n_since_checkpoint = 0
    groups = list(scoreable.groupby(["lang", "bucket"], observed=True))
    for (lang, bucket), grp in groups:
        n_new_this_group = 0
        for _, r in grp.iterrows():
            key = (r.segment_id, r.bucket, r.seed)
            if key in done_keys:
                continue
            src_ids, ref_ids = get_canon(r.segment_id)
            score = MP.metricx_score_from_ids(src_ids, list(r.mt_ids), ref_ids)
            mqm = seg_by_id_cond.loc[(r.segment_id, "native"), "mqm_score"]
            results.append({"segment_id": r.segment_id, "lang": lang, "direction": "forward",
                            "bucket": bucket, "seed": r.seed, "ratio": r.ratio,
                            "metricx24": score, "mqm_score": mqm})
            done_keys.add(key)
            n_new_this_group += 1
            n_since_checkpoint += 1
            if n_since_checkpoint >= 500:
                pd.DataFrame(results).to_csv(out_path, index=False)
                n_since_checkpoint = 0
        elapsed = time.time() - t0
        print(f"scored {lang} {bucket}: {n_new_this_group} new (of {len(grp)}), "
              f"running total {len(results)}, elapsed {elapsed/60:.1f} min", flush=True)
        pd.DataFrame(results).to_csv(out_path, index=False)

    res_df = pd.DataFrame(results)
    res_df.to_csv(out_path, index=False)

    print("\n" + "=" * 78)
    print("Dose-response: mean MetricX-24 and Spearman(MetricX,MQM) per bucket, per language")
    print("MetricX-24 is error-scale: rho is NEGATIVE throughout if tracking quality correctly; "
          "'degrades' = |rho| decreasing, never sign-flipped.")
    print("=" * 78)
    curve_rows = []
    for lang in C.LANG_ORDER:
        sub_dir = res_df[res_df.lang == lang]
        for bucket in sub_dir.bucket.unique():
            sub = sub_dir[sub_dir.bucket == bucket]
            if len(sub) < 4:
                continue
            mean_ratio = sub.ratio.mean()
            mean_metricx = sub.metricx24.mean()
            rho = sps.spearmanr(sub.metricx24, sub.mqm_score)[0] if sub.metricx24.nunique() > 1 else float("nan")
            curve_rows.append({"lang": lang, "direction": "forward", "bucket": bucket,
                               "n": len(sub), "mean_ratio": mean_ratio,
                               "mean_metricx24": mean_metricx, "spearman_metricx_mqm": rho,
                               "abs_spearman_metricx_mqm": abs(rho) if rho == rho else float("nan")})
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(C.OUT_TABLES / "reseg_dose_response_mt5.csv", index=False)
    print(curve_df.sort_values(["lang", "mean_ratio"]).to_string(index=False))

    print("\n" + "=" * 78)
    print("Monotonicity check: does |Spearman(MetricX,MQM)| DECREASE as ratio increases? "
          "(replication test -- COMET/XLM-R showed Spearman(ratio,rho) in [-0.95,-1.00])")
    print("=" * 78)
    mono_rows = []
    for lang in C.LANG_ORDER:
        sub = curve_df[curve_df.lang == lang].sort_values("mean_ratio").dropna(subset=["abs_spearman_metricx_mqm"])
        if len(sub) < 3:
            mono_rows.append({"lang": lang, "spearman_ratio_vs_abs_rho": None, "n": len(sub)})
            continue
        corr = sps.spearmanr(sub.mean_ratio, sub.abs_spearman_metricx_mqm)[0]
        mono_rows.append({"lang": lang, "spearman_ratio_vs_abs_rho": corr, "n": len(sub)})
        print(f"{lang}: Spearman(ratio, |rho_metricx_mqm|) across buckets = {corr:.3f} "
              f"(negative = |agreement| degrades with fragmentation, replicating the XLM-R pattern)")
    mono_df = pd.DataFrame(mono_rows)
    mono_df.to_csv(C.OUT_TABLES / "reseg_monotonicity_mt5.csv", index=False)

    n_replicating = sum(1 for r in mono_rows if r["spearman_ratio_vs_abs_rho"] is not None
                        and r["spearman_ratio_vs_abs_rho"] < 0)
    print(f"\nREPLICATION VERDICT: {n_replicating}/5 languages show the same-direction "
          f"(monotone collapse) pattern as XLM-R/COMET-22.")

    try:
        PROV.register(
            "reseg_dose_response_mt5", {
                "n_scored": len(res_df),
                "dose_response_curve": curve_df.to_dict("records"),
                "monotonicity_per_lang": mono_rows,
                "n_languages_replicating": n_replicating,
            },
            PROV.stamp(Path(__file__), modules=["metrics_panel", "noncanonical_mt5", "tokenisers", "constants"],
                       seed=C.SEED,
                       note="re-segmentation dose-response, MetricX-24 (mT5 vocabulary), forward direction only"))
    except Exception as exc:
        print(f"(non-fatal: PROV.register failed: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

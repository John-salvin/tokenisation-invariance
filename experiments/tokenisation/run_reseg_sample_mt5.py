#!/usr/bin/env python3
"""Re-segmentation sampling with the mT5 vocabulary (CPU).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import noncanonical as NC
import noncanonical_mt5 as NCM

TOKENISER = "mt5"
N_SEGMENTS_PER_LANG = 300
SEEDS = [0, 1, 2, 3, 4]


def _sample_forward(text: str, seg_id: str, lang: str) -> tuple[list[dict], int, int]:
    rows, n_attempted, n_excluded = [], 0, 0
    for lo, hi in NC.RATIO_BUCKETS:
        bucket_label = f"[{lo:.2f},{hi:.2f})"
        for seed in SEEDS:
            n_attempted += 1
            try:
                r = NCM.sample_bucket_mt5(text, lo, hi, random.Random(seed), TOKENISER)
                if r["reachable"]:
                    NCM.assert_round_trip_mt5(text, r["ids"], TOKENISER)
                    rows.append({"segment_id": seg_id, "lang": lang, "direction": "forward",
                                "bucket": bucket_label, "seed": seed,
                                "canonical_k": r["canonical_k"], "total_k": r["total_k"],
                                "ratio": r["ratio"], "mt_ids": r["ids"]})
                else:
                    rows.append({"segment_id": seg_id, "lang": lang, "direction": "forward",
                                "bucket": bucket_label, "seed": seed,
                                "canonical_k": r["canonical_k"], "total_k": None,
                                "ratio": None, "mt_ids": None})
            except NC.NoncanonicalError:
                n_excluded += 1
    return rows, n_attempted, n_excluded


def main() -> int:
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(Path.home() / "tokinv_work/repo/panel-pipeline/data/interim/segments.parquet")
    seg = seg[seg.mqm_valid].copy()
    seg["target"] = seg["target"].astype(str).map(NC.normalize_text_for_reseg)

    all_rows: list[dict] = []
    n_seg_excluded_forward = 0
    stats_rows = []

    for lang in C.LANG_ORDER:
        nat = seg[(seg.lang == lang) & (seg.condition == "native")]
        nat = nat.sample(n=min(N_SEGMENTS_PER_LANG, len(nat)), random_state=C.SEED)

        n_fwd_attempted = n_fwd_excluded = 0
        for _, r in nat.iterrows():
            try:
                rows, na, ne = _sample_forward(r.target, r.segment_id, lang)
                all_rows.extend(rows)
                n_fwd_attempted += na
                n_fwd_excluded += ne
            except NC.NoncanonicalError as exc:
                n_seg_excluded_forward += 1
                print(f"[forward exclude] {lang} {r.segment_id}: {exc}")

        stats_rows.append({"lang": lang, "n_native_segments": len(nat),
                           "n_forward_cells_attempted": n_fwd_attempted,
                           "n_forward_cells_excluded": n_fwd_excluded})
        print(f"{lang}: native={len(nat)} forward_excluded={n_fwd_excluded}/{n_fwd_attempted}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(C.DATA_INTERIM / "reseg_samples_mt5.parquet", index=False)
    pd.DataFrame(stats_rows).to_csv(C.OUT_TABLES / "reseg_sample_stats_mt5.csv", index=False)

    print(f"\nTotal sample rows (including unreachable-bucket placeholders): {len(df)}")
    print(f"Reachable (scoreable) rows: {df.mt_ids.notna().sum()}")
    print(f"Segments excluded entirely (whole-text canonical round-trip failure): "
          f"forward={n_seg_excluded_forward} / {sum(s['n_native_segments'] for s in stats_rows)} attempted "
          f"(expected ~2.2% from the full-corpus check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

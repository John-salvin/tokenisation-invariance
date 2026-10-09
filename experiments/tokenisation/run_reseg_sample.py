#!/usr/bin/env python3
"""Re-segmentation sampling for IndicMT Eval (CPU).

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
import tokenisers as T

TOKENISER = "xlmr"
N_SEGMENTS_PER_LANG = 300
SEEDS = [0, 1, 2, 3, 4]


def _sample_forward(text: str, seg_id: str, lang: str) -> tuple[list[dict], int, int]:
    rows, n_attempted, n_excluded = [], 0, 0
    for lo, hi in NC.RATIO_BUCKETS:
        bucket_label = f"[{lo:.2f},{hi:.2f})"
        for seed in SEEDS:
            n_attempted += 1
            try:
                r = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(seed))
                if r["reachable"]:
                    NC.assert_round_trip(text, r["ids"], TOKENISER)
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
    n_attempted += 1
    try:
        r = NC.sample_char_level(text, TOKENISER)
        if r["reachable"]:
            NC.assert_round_trip(text, r["ids"], TOKENISER)
            rows.append({"segment_id": seg_id, "lang": lang, "direction": "forward",
                        "bucket": NC.CHAR_LEVEL_BUCKET, "seed": 0,
                        "canonical_k": r["canonical_k"], "total_k": r["total_k"],
                        "ratio": r["ratio"], "mt_ids": r["ids"]})
        else:
            rows.append({"segment_id": seg_id, "lang": lang, "direction": "forward",
                        "bucket": NC.CHAR_LEVEL_BUCKET, "seed": 0,
                        "canonical_k": r["canonical_k"], "total_k": None,
                        "ratio": None, "mt_ids": None})
    except NC.NoncanonicalError:
        n_excluded += 1
    return rows, n_attempted, n_excluded


def _sample_reverse(text: str, seg_id: str, lang: str) -> tuple[list[dict], int, int]:
    rows, n_attempted, n_excluded = [], 0, 0
    for lo, hi in NC.REVERSE_RATIO_BUCKETS:
        bucket_label = f"[{lo:.2f},{hi:.2f})"
        for seed in SEEDS:
            n_attempted += 1
            try:
                r = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(seed),
                                     inclusive_hi=False)
                if r["reachable"]:
                    NC.assert_round_trip(text, r["ids"], TOKENISER)
                    rows.append({"segment_id": seg_id, "lang": lang, "direction": "reverse",
                                "bucket": bucket_label, "seed": seed,
                                "canonical_k": r["canonical_k"], "total_k": r["total_k"],
                                "ratio": r["ratio"], "mt_ids": r["ids"]})
                else:
                    rows.append({"segment_id": seg_id, "lang": lang, "direction": "reverse",
                                "bucket": bucket_label, "seed": seed,
                                "canonical_k": r["canonical_k"], "total_k": None,
                                "ratio": None, "mt_ids": None})
            except NC.NoncanonicalError:
                n_excluded += 1
    n_attempted += 1
    try:
        r = NC.sample_min_floor(text, TOKENISER, random.Random(0))
        if r["reachable"]:
            NC.assert_round_trip(text, r["ids"], TOKENISER)
            rows.append({"segment_id": seg_id, "lang": lang, "direction": "reverse",
                        "bucket": NC.MIN_FLOOR_BUCKET, "seed": 0,
                        "canonical_k": r["canonical_k"], "total_k": r["total_k"],
                        "ratio": r["ratio"], "mt_ids": r["ids"]})
        else:
            rows.append({"segment_id": seg_id, "lang": lang, "direction": "reverse",
                        "bucket": NC.MIN_FLOOR_BUCKET, "seed": 0,
                        "canonical_k": r["canonical_k"], "total_k": None,
                        "ratio": None, "mt_ids": None})
    except NC.NoncanonicalError:
        n_excluded += 1
    return rows, n_attempted, n_excluded


def main() -> int:
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid].copy()
    seg["target"] = seg["target"].astype(str).map(NC.normalize_text_for_reseg)

    all_rows: list[dict] = []
    n_seg_excluded_forward = 0
    n_seg_excluded_reverse = 0
    stats_rows = []

    for lang in C.LANG_ORDER:
        nat = seg[(seg.lang == lang) & (seg.condition == "native")]
        rom = seg[(seg.lang == lang) & (seg.condition == "romanised")]
        nat = nat.sample(n=min(N_SEGMENTS_PER_LANG, len(nat)), random_state=C.SEED)
        rom = rom[rom.segment_id.isin(nat.segment_id)]

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

        n_rev_attempted = n_rev_excluded = 0
        for _, r in rom.iterrows():
            try:
                rows, na, ne = _sample_reverse(r.target, r.segment_id, lang)
                all_rows.extend(rows)
                n_rev_attempted += na
                n_rev_excluded += ne
            except NC.NoncanonicalError as exc:
                n_seg_excluded_reverse += 1
                print(f"[reverse exclude] {lang} {r.segment_id}: {exc}")

        stats_rows.append({"lang": lang, "n_native_segments": len(nat), "n_romanised_segments": len(rom),
                           "n_forward_cells_attempted": n_fwd_attempted,
                           "n_forward_cells_excluded": n_fwd_excluded,
                           "n_reverse_cells_attempted": n_rev_attempted,
                           "n_reverse_cells_excluded": n_rev_excluded})
        print(f"{lang}: native={len(nat)} romanised={len(rom)} "
              f"forward_excluded={n_fwd_excluded}/{n_fwd_attempted} "
              f"reverse_excluded={n_rev_excluded}/{n_rev_attempted}")

    df = pd.DataFrame(all_rows)
    df.to_parquet(C.DATA_INTERIM / "reseg_samples.parquet", index=False)
    pd.DataFrame(stats_rows).to_csv(C.OUT_TABLES / "reseg_sample_stats.csv", index=False)

    print(f"\nTotal sample rows (including unreachable-bucket placeholders): {len(df)}")
    print(f"Reachable (scoreable) rows: {df.mt_ids.notna().sum()}")
    print(f"Segments excluded entirely (canonical round-trip failure at the whole-text level): "
          f"forward={n_seg_excluded_forward} reverse={n_seg_excluded_reverse}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

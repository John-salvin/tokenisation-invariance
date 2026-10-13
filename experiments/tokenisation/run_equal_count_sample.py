#!/usr/bin/env python3
"""Re-segmentations within five percent of the canonical token count (CPU).

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

TOKENISER = "xlmr"
SEEDS = [0, 1, 2, 3, 4]
BUCKETS = [(0.95, 1.00, False), (1.00, 1.05, True)]


def bucket_label(lo: float, hi: float, inclusive_hi: bool) -> str:
    close = "]" if inclusive_hi else ")"
    return f"[{lo:.2f},{hi:.2f}{close}"


def sample_segment(text: str, seg_id: str, lang: str) -> tuple[list[dict], int, int]:
    rows, n_attempted, n_excluded = [], 0, 0
    for lo, hi, inclusive_hi in BUCKETS:
        label = bucket_label(lo, hi, inclusive_hi)
        for seed in SEEDS:
            n_attempted += 1
            try:
                r = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(seed),
                                     inclusive_hi=inclusive_hi)
                if r["reachable"]:
                    NC.assert_round_trip(text, r["ids"], TOKENISER)
                    rows.append({"segment_id": seg_id, "lang": lang, "bucket": label,
                                "seed": seed, "canonical_k": r["canonical_k"],
                                "total_k": r["total_k"], "ratio": r["ratio"], "mt_ids": r["ids"]})
                else:
                    rows.append({"segment_id": seg_id, "lang": lang, "bucket": label,
                                "seed": seed, "canonical_k": r["canonical_k"],
                                "total_k": None, "ratio": None, "mt_ids": None})
            except NC.NoncanonicalError:
                n_excluded += 1
    return rows, n_attempted, n_excluded


def main() -> int:
    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[(seg.mqm_valid) & (seg.condition == "native")].copy()
    seg["target"] = seg["target"].astype(str).map(NC.normalize_text_for_reseg)

    all_rows: list[dict] = []
    stats_rows = []
    for lang in C.LANG_ORDER:
        sub = seg[seg.lang == lang]
        n_seg_excluded = 0
        n_attempted_total = n_excluded_total = 0
        for _, r in sub.iterrows():
            rows, n_att, n_exc = sample_segment(r["target"], r["segment_id"], lang)
            if n_exc == n_att:
                n_seg_excluded += 1
            all_rows.extend(rows)
            n_attempted_total += n_att
            n_excluded_total += n_exc
        stats_rows.append({"lang": lang, "n_segments": len(sub),
                           "n_segments_fully_excluded": n_seg_excluded,
                           "n_attempted": n_attempted_total, "n_excluded": n_excluded_total,
                           "pct_excluded": 100.0 * n_excluded_total / n_attempted_total})
        print(f"{lang}: {len(sub)} segments, "
              f"{n_attempted_total - n_excluded_total}/{n_attempted_total} "
              f"attempts survived round-trip exclusion")

    df = pd.DataFrame(all_rows)
    df.to_parquet(C.DATA_INTERIM / "equal_count_samples.parquet", index=False)
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(C.OUT_TABLES / "equal_count_sample_stats.csv", index=False)

    reach = (df[df.mt_ids.notna()]
            .groupby(["lang", "bucket"], observed=True).size()
            .reset_index(name="n_reachable"))
    total = df.groupby(["lang", "bucket"], observed=True).size().reset_index(name="n_total")
    reach = reach.merge(total, on=["lang", "bucket"], how="right").fillna({"n_reachable": 0})
    reach["pct_reachable"] = 100.0 * reach["n_reachable"] / reach["n_total"]
    reach.to_csv(C.OUT_TABLES / "equal_count_bucket_reachability.csv", index=False)
    print(reach.to_string(index=False))
    print(f"\nWrote {len(df)} rows to data/interim/equal_count_samples.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())

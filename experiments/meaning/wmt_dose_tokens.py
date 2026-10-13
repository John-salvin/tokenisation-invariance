#!/usr/bin/env python3
"""XLM-R token counts of the Section 6 conditions beyond Indic (lossy and the
four noise levels; experiments/meaning/wmt_dose_conditions.py), so that the
no-truncation rule (no row a metric would truncate) can be applied to them as to native and romanised.
Uses the pinned tokeniser of tokinv.tokenisers; --check first compares it with
the counts stored with the native and romanised scores.

    python experiments/meaning/wmt_dose_tokens.py --dose artifacts/wmt_dose \
        --out data/derived/beyond_indic [--check]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tokinv.tokenisers import get

CONDS = ["lossy", "noise02", "noise05", "noise10", "noise20"]


def counts(texts) -> list[int]:
    tok = get("xlmr")
    return [len(x) for x in tok([str(t) for t in texts], add_special_tokens=False)["input_ids"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    for f in sorted(a.dose.glob("*.parquet")):
        pair = f.stem
        d = pd.read_parquet(f)
        if a.check:
            c = pd.read_parquet(a.out / f"{pair}_tokens.parquet").merge(
                d[["system", "seg_idx", "mt_native", "mt_rom", "ref_rom"]], on=["system", "seg_idx"])
            ok = all(counts(c[t]) == c[k].tolist() for t, k in
                     [("mt_native", "ntok_mt_native"), ("mt_rom", "ntok_mt_romanised"), ("ref_rom", "ntok_ref_romanised")])
            print(pair, "matches cluster counts:", ok, flush=True)
            if not ok:
                raise SystemExit("tokeniser mismatch")
        out = d[["system", "seg_idx"]].copy()
        for c in CONDS:
            out[f"ntok_mt_{c}"] = counts(d[f"mt_{c}"])
            out[f"ntok_ref_{c}"] = counts(d[f"ref_{c}"])
        out.to_parquet(a.out / f"{pair}_dose_tokens.parquet", index=False)
        print("wrote", pair, len(out), flush=True)


if __name__ == "__main__":
    main()

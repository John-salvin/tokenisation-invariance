#!/usr/bin/env python3
"""Intrinsic tokeniser measures on IndicMT Eval for the tokeniser panel.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import intrinsic as I
import tokenisers as T


def main() -> int:
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    print(f"loaded {len(seg):,} rows")

    print("\n=== GATE: reproduce committed workbook token counts ===")
    rep = T.verify_against_workbook(seg)
    print(rep.to_string(index=False))
    print("gate passed\n")

    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    rep.to_csv(C.OUT_TABLES / "tokeniser_gate.csv", index=False)

    per_seg, mattrs, failed = [], [], {}
    for tok in T.PANEL:
        t0 = time.time()
        try:
            per_seg.append(I.per_segment(seg, tok))
            mattrs.append(I.corpus_mattr(seg, tok))
            print(f"  [{tok:10s}] ok  {time.time()-t0:6.1f}s")
        except Exception as exc:
            print(f"  [{tok:10s}] FAILED: {type(exc).__name__}: {exc}")
            failed[tok] = repr(exc)

    if failed:
        raise SystemExit(
            f"\n{len(failed)} tokeniser(s) failed: {list(failed)}\n"
            "Refusing to write a half-populated panel (design rule)."
        )

    ps = pd.concat(per_seg, ignore_index=True)
    agg = I.aggregate(ps)
    mat = pd.concat(mattrs, ignore_index=True)

    C.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    ps.to_parquet(C.DATA_INTERIM / "intrinsic_per_segment.parquet", index=False)
    agg.to_csv(C.OUT_TABLES / "intrinsic_indicmt.csv", index=False)
    mat.to_csv(C.OUT_TABLES / "mattr_indicmt.csv", index=False)

    (C.OUT_TABLES / "tokeniser_panel_manifest.json").write_text(
        json.dumps(T.panel_manifest(), indent=2))

    print(f"\nwrote intrinsic_indicmt.csv ({len(agg)} rows), mattr_indicmt.csv")
    print("\n=== TP by tokeniser x condition (fixed language order) ===")
    piv = agg.pivot_table(index="tokeniser", columns=["lang", "condition"],
                          values="tp", observed=True)
    print(piv.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

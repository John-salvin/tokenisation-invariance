#!/usr/bin/env python3
"""Regenerate data/derived/adequacy/iso_dose_segments.parquet (Section 6).

Every condition is placed on one measured x-axis, the round-trip character
error rate: romanise, transliterate back to the native script, and take the
CER against the original native string. It is measured for ISO 15919, for
each synthetic-noise level, and for IndicXlit alike, never assumed.

Inputs
  --iso      per-segment ISO 15919 conditions with the noise variants
             (experiments/meaning/iso_romanise_noise.py)
  --scores   COMET-22 on each ISO condition
             (experiments/meaning/iso_score.py)
  --derom    IndicXlit de-romanisation of the romanised targets
             (experiments/meaning/deromanise_indicxlit.py)
and data/indic/segments.parquet for the native and IndicXlit COMET-22 scores.

Output: one row per segment with mqm, every condition's COMET-22 and every
condition's round-trip CER. The notebooks (tokinv.adequacy) need nothing else.

    python experiments/meaning/build_adequacy_inputs.py --iso ISO.parquet \
        --scores ISO_SCORES.csv --derom DEROM.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jiwer
import pandas as pd
from aksharamukha import transliterate as tr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "derived" / "adequacy" / "iso_dose_segments.parquet"
SCRIPT = {"hin": "Devanagari", "mar": "Devanagari", "guj": "Gujarati",
          "tam": "Tamil", "mal": "Malayalam"}
NOISE = ["02", "05", "10", "20"]


def cer(a, b):
    try:
        return float(jiwer.cer(a, b)) if a else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", required=True, type=Path)
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--derom", required=True, type=Path)
    a = ap.parse_args()

    iso = pd.read_parquet(a.iso)
    sc = pd.read_csv(a.scores)
    iso = iso.merge(sc[["segment_id"] + [c for c in sc.columns if c.startswith("comet22_")]],
                    on="segment_id", how="left")
    seg = pd.read_parquet(ROOT / "data" / "indic" / "segments.parquet")
    seg["condition"] = seg["condition"].astype(str)
    nat = seg[seg.condition == "native"][["segment_id", "comet_22"]].rename(
        columns={"comet_22": "comet22_native"})
    rom = seg[seg.condition == "romanised"][["segment_id", "comet_22"]].rename(
        columns={"comet_22": "comet22_indicxlit"})
    iso = iso.merge(nat, on="segment_id", how="left").merge(rom, on="segment_id", how="left")

    der = pd.read_parquet(a.derom)
    iso = iso.merge(der[["segment_id", "derom_target"]], on="segment_id", how="left")
    iso["cer_indicxlit"] = [cer(x, y) for x, y in zip(iso.target_native, iso.derom_target)]
    for lv in NOISE:
        back = []
        for lang, sub in iso.groupby("lang"):
            back.append(pd.Series([tr.process("ISO", SCRIPT[lang], t)
                                   for t in sub[f"target_noise{lv}"]], index=sub.index))
        back = pd.concat(back).sort_index()
        iso[f"cer_noise{lv}"] = [cer(x, y) for x, y in zip(iso.target_native, back)]

    keep = (["segment_id", "lang", "system", "source_en", "mqm_score", "mqm_valid",
             "target_iso", "cer_target", "cer_reference", "cer_indicxlit"]
            + [f"cer_noise{lv}" for lv in NOISE]
            + ["comet22_native", "comet22_iso"] + [f"comet22_iso_noise{lv}" for lv in NOISE]
            + ["comet22_indicxlit"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    iso[keep].to_parquet(OUT, index=False)
    print(f"wrote {OUT} ({len(iso)} rows)")


if __name__ == "__main__":
    main()

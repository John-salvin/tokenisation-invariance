#!/usr/bin/env python3
"""ISO 15919 round-trip error with and without danda normalisation, per
Indic language (Appendix: Distortion at Zero Information Loss, the Marathi
note). The round trip is exactly that of
experiments/meaning/iso_romanise_noise.py (aksharamukha native -> ISO -> native, jiwer CER against the
native target); its unnormalised mean reproduces the stored cer_target.
Normalising maps the danda and double danda to '.' on both sides.

    python experiments/meaning/iso_danda_cer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
from aksharamukha import transliterate as tr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tokinv.paths import DATA, INDIC  # noqa: E402

SCRIPT = {"hin": "Devanagari", "mar": "Devanagari", "guj": "Gujarati", "tam": "Tamil", "mal": "Malayalam"}
OUT = DATA / "derived" / "adequacy" / "iso_cer_danda.csv"


def cer(a: str, b: str) -> float:
    return float(jiwer.cer(a, b)) if a else float("nan")


def norm(s: str) -> str:
    return s.replace("॥", ".").replace("।", ".")


def main() -> None:
    seg = pd.read_parquet(INDIC / "segments.parquet")
    rows = []
    for lang, sub in seg[seg.condition == "native"].groupby("lang"):
        raw, nrm = [], []
        for t in sub.target:
            back = tr.process("ISO", SCRIPT[lang], tr.process(SCRIPT[lang], "ISO", t))
            raw.append(cer(t, back))
            nrm.append(cer(norm(t), norm(back)))
        rows.append({"lang": lang, "n": len(sub), "mean_cer": float(np.nanmean(raw)),
                     "mean_cer_danda_normalised": float(np.nanmean(nrm))})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).round(5).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Section 6 conditions beyond Indic: for each romanised WMT pair,
add a lossy romanisation and four synthetic-noise levels of the lossless one,
with each condition's round-trip character error rate MEASURED.

  lossless   ISO 15919 / ISO 9 / ISO 233 (experiments/data_prep/romanise_wmt.py)
  lossy      Devanagari: aksharamukha RomanReadable; Cyrillic and Arabic: the
             lossless form with diacritics and modifier letters removed
  noiseXX    the lossless form with XX percent of non-space characters
             replaced or deleted, target and reference alike, using the
             procedure and seed of the IndicMT Eval experiment (iso_romanise_noise)

Error rate: transliterate each condition back to the native script and take
the CER against the lossless form transliterated back the same way. WMT text
contains original Latin words, which any whole-text inverse also alters;
measuring against the lossless round trip removes that common term, so the
lossless condition has CER 0 by construction and the others are comparable.

    python experiments/meaning/wmt_dose_conditions.py --inp artifacts/wmt_romanised --out artifacts/wmt_dose
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import romanise_wmt as R  # noqa: E402

SEED = 20260919                       # as in the IndicMT Eval experiment
NOISE = [0.02, 0.05, 0.10, 0.20]


def noise_text(s, target_cer, rng, alphabet):
    """Verbatim from experiments/meaning/iso_romanise_noise.py."""
    idx = [i for i, c in enumerate(s) if not c.isspace()]
    if not idx or target_cer <= 0:
        return s
    k = min(int(round(target_cer * len(s))), len(idx))
    if k == 0:
        return s
    pos = rng.choice(len(idx), size=k, replace=False)
    chars, drop = list(s), set()
    for p in pos:
        i = idx[p]
        if rng.random() < 0.5:
            chars[i] = alphabet[int(rng.integers(len(alphabet)))]
        else:
            drop.add(i)
    return "".join(c for j, c in enumerate(chars) if j not in drop)


def fold(text: str) -> str:
    """Drop diacritics and modifier letters: the lossy Cyrillic/Arabic form."""
    d = unicodedata.normalize("NFD", text)
    d = "".join(c for c in d if not unicodedata.combining(c))
    return d.translate({ord(c): None for c in "ʹʺʾʿˊ˝°ⁿ_"})


def lossy(text: str, script: str) -> str:
    if script == "Devanagari":
        from aksharamukha import transliterate as tr
        lo, hi = R.BLOCKS[script]
        out, i = [], 0
        while i < len(text):
            j = i
            native = lo <= ord(text[i]) <= hi
            while j < len(text) and (lo <= ord(text[j]) <= hi) == native:
                j += 1
            out.append(tr.process("Devanagari", "RomanReadable", text[i:j]) if native else text[i:j])
            i = j
        return "".join(out)
    lossless, _, _ = R.romanise(text, script)
    return fold(lossless)


def back(text: str, script: str, scheme: str) -> str:
    """Whole-text transliteration back to the native script."""
    if script == "Devanagari":
        from aksharamukha import transliterate as tr
        return unicodedata.normalize("NFD", tr.process(scheme, "Devanagari", text))
    _, rev = R.CYR if script == "Cyrillic" else R.ARB
    return R._invert(text, rev)


def cer(a: str, b: str) -> float:
    return float(jiwer.cer(a, b)) if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", nargs="*", default=list(R.PAIRS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for pair in a.pairs:
        script = R.PAIRS[pair][2]
        d = pd.read_parquet(a.inp / f"{pair}.parquet")
        d = d[(d.mt_native_chars > 0) & d.roundtrip_ok].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        alpha = sorted({ch for s in d.mt_rom.head(2000) for ch in s if not ch.isspace()})
        lossless_scheme = "ISO" if script == "Devanagari" else None
        base = [back(t, script, lossless_scheme) for t in d.mt_rom]
        d["mt_lossy"] = [lossy(t, script) for t in d.mt_native]
        d["ref_lossy"] = [lossy(t, script) for t in d.ref_native]
        d["cer_lossless"] = [cer(b, back(t, script, lossless_scheme)) for b, t in zip(base, d.mt_rom)]
        d["cer_lossy"] = [cer(b, back(t, script, "RomanReadable" if script == "Devanagari" else None))
                          for b, t in zip(base, d.mt_lossy)]
        for lv in NOISE:
            tag = f"noise{int(lv * 100):02d}"
            d[f"mt_{tag}"] = [noise_text(s, lv, rng, alpha) for s in d.mt_rom]
            d[f"ref_{tag}"] = [noise_text(s, lv, rng, alpha) for s in d.ref_rom]
            d[f"cer_{tag}"] = [cer(b, back(t, script, lossless_scheme)) for b, t in zip(base, d[f"mt_{tag}"])]
        d.to_parquet(a.out / f"{pair}.parquet", index=False)
        cers = {c: round(float(d[c].mean()), 4) for c in d.columns if c.startswith("cer_")}
        print(f"{pair:10s} rows={len(d)} {cers}", flush=True)


if __name__ == "__main__":
    main()

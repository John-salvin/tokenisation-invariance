#!/usr/bin/env python3
"""Romanise WMT targets and references for the Script-Invariance experiments
beyond Indic: Devanagari (Hindi, Bhojpuri), Cyrillic (Russian,
Ukrainian) and Arabic.

Every scheme is letter-for-letter and reversible, and every segment's round
trip back to the native script is checked; a segment that does not round-trip
exactly is excluded and counted, never scored.

  Devanagari  ISO 15919 (aksharamukha), the scheme already used for IndicMT Eval
  Cyrillic    ISO 9:1995 (one Latin letter, with diacritics, per Cyrillic letter)
  Arabic      ISO 233-style letter mapping (unvocalised text stays unvocalised)

Writes one parquet per pair: system, seg_idx, human, src, ref_native,
mt_native, ref_rom, mt_rom, roundtrip_ok.

    python experiments/data_prep/romanise_wmt.py --mtme DIR --out DIR [--pairs ...]
"""
from __future__ import annotations

import argparse
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

PAIRS = {  # pair: (testset, human-score file kind, script)
    "en-hi": ("wmt24", "esa", "Devanagari"), "en-bho_IN": ("wmt25", "esa-merged", "Devanagari"),
    "en-ru": ("wmt24", "esa", "Cyrillic"), "en-ru_RU": ("wmt25", "esa-merged", "Cyrillic"),
    "en-uk": ("wmt24", "esa", "Cyrillic"), "en-uk_UA": ("wmt25", "esa-merged", "Cyrillic"),
    "en-ar_EG": ("wmt25", "esa-merged", "Arabic"),
}

# ISO 9:1995, Russian and Ukrainian letters (system A, one-to-one)
ISO9 = {"а": "a", "б": "b", "в": "v", "г": "g", "ґ": "g̀", "д": "d", "е": "e", "ё": "ë",
        "є": "ê", "ж": "ž", "з": "z", "и": "i", "і": "ì", "ї": "ï", "й": "j", "к": "k",
        "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "ш": "š", "щ": "ŝ", "ъ": "ʺ",
        "ы": "y", "ь": "ʹ", "э": "è", "ю": "û", "я": "â"}
# ISO 233-style: each Arabic letter to one distinct Latin symbol
ISO233 = {"ء": "ʾ", "آ": "â", "أ": "ạ", "ؤ": "ẉ", "إ": "ị", "ئ": "ỵ", "ٱ": "ȁ", "ا": "ā", "ب": "b",
          "ة": "ẗ", "ت": "t", "ث": "ṯ", "ج": "ǧ", "ح": "ḥ", "خ": "ẖ", "د": "d", "ذ": "ḏ",
          "ر": "r", "ز": "z", "س": "s", "ش": "š", "ص": "ṣ", "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ",
          "ع": "ʿ", "غ": "ġ", "ف": "f", "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
          "ه": "h", "و": "w", "ى": "ỳ", "ي": "y", "ـ": "_", "َ": "a", "ُ": "u", "ِ": "i",
          "ً": "aⁿ", "ٌ": "uⁿ", "ٍ": "iⁿ", "ّ": "̃", "ْ": "°", "،": ",", "؛": ";", "؟": "?",
          "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7",
          "٨": "8", "٩": "9"}


# ISO 9 writes hard and soft signs the same in both cases; give the rare
# capitals their own symbols so the mapping stays reversible
CYR_UPPER_SIGNS = {"Ъ": "˝", "Ь": "ˊ"}


def _table(m: dict[str, str], upper: bool) -> tuple[dict, dict]:
    fwd = dict(m)
    if upper:
        fwd.update({k.upper(): v[0].upper() + v[1:] for k, v in m.items()
                    if k.upper() != k and k.upper() not in CYR_UPPER_SIGNS})
        fwd.update(CYR_UPPER_SIGNS)
    rev = {v: k for k, v in fwd.items()}
    assert len(rev) == len(fwd), "transliteration table is not one-to-one"
    return fwd, rev


def _apply(text: str, fwd: dict) -> str:
    return "".join(fwd.get(ch, ch) for ch in text)


def _invert(text: str, rev: dict) -> str:
    keys = sorted(rev, key=len, reverse=True)   # longest match first
    out, i = [], 0
    while i < len(text):
        for k in keys:
            if text.startswith(k, i):
                out.append(rev[k])
                i += len(k)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


CYR = _table(ISO9, upper=True)
ARB = _table(ISO233, upper=False)


BLOCKS = {"Devanagari": (0x0900, 0x097F), "Cyrillic": (0x0400, 0x04FF), "Arabic": (0x0600, 0x06FF)}


@lru_cache(maxsize=None)
def _span_romanise(span: str, script: str) -> tuple[str, bool]:
    """Cached: the same word runs recur thousands of times across systems."""
    if script == "Devanagari":
        from aksharamukha import transliterate as tr
        out = tr.process("Devanagari", "ISO", span)
        # equality up to Unicode canonical equivalence: the inverse returns
        # precomposed nukta letters (U+095C) where the source has the
        # decomposed pair (U+0921 U+093C); the two are the same character
        nfd = lambda s: unicodedata.normalize("NFD", s)
        return out, nfd(tr.process("ISO", "Devanagari", out)) == nfd(span)
    fwd, rev = CYR if script == "Cyrillic" else ARB
    out = _apply(span, fwd)
    return out, _invert(out, rev) == span


def romanise(text: str, script: str) -> tuple[str, bool, int]:
    """Romanise only the runs of native-script characters; text already in
    another script (names, URLs, English words) is left exactly as it is.
    Returns the romanised text, whether every run round-trips exactly, and
    the number of native-script characters converted."""
    lo, hi = BLOCKS[script]
    native = lambda ch: lo <= ord(ch) <= hi or (script == "Arabic" and ch in "،؛؟")
    out, ok, n, i = [], True, 0, 0
    while i < len(text):
        j = i
        if native(text[i]):
            while j < len(text) and native(text[j]):
                j += 1
            r, good = _span_romanise(text[i:j], script)
            out.append(r)
            ok &= good and not any(native(c) for c in r)
            n += j - i
        else:
            while j < len(text) and not native(text[j]):
                j += 1
            out.append(text[i:j])
        i = j
    return "".join(out), ok, n


def load_pair(mtme: Path, pair: str) -> pd.DataFrame:
    ts, kind, _ = PAIRS[pair]
    d = mtme / ts
    src = [l.rstrip("\n") for l in open(d / "sources" / f"{pair}.txt", encoding="utf-8")]
    ref = [l.rstrip("\n") for l in open(d / "references" / f"{pair}.refA.txt", encoding="utf-8")]
    by = {}
    for line in open(d / "human-scores" / f"{pair}.{kind}.seg.score", encoding="utf-8"):
        s, v = line.rstrip("\n").split("\t")
        by.setdefault(s, []).append(v)
    rows = []
    for s in sorted(by):
        if s.lower().startswith("ref") or all(x == "None" for x in by[s]):
            continue
        out = [l.rstrip("\n") for l in open(d / "system-outputs" / pair / f"{s}.txt", encoding="utf-8")]
        for i, (mt, v) in enumerate(zip(out, by[s])):
            if v != "None" and mt.strip():
                rows.append({"system": s, "seg_idx": i, "human": float(v), "src": src[i],
                             "ref_native": ref[i], "mt_native": mt})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtme", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", nargs="*", default=list(PAIRS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for pair in a.pairs:
        script = PAIRS[pair][2]
        d = load_pair(a.mtme, pair)
        mt = [romanise(t, script) for t in d.mt_native]
        rf = [romanise(t, script) for t in d.ref_native]
        d["mt_rom"] = [x[0] for x in mt]
        d["ref_rom"] = [x[0] for x in rf]
        d["roundtrip_ok"] = [a_[1] and b_[1] for a_, b_ in zip(mt, rf)]
        # rows with no native script in the translation (e.g. WMT's canary
        # lines) are identical in both conditions and carry no signal
        d["mt_native_chars"] = [x[2] for x in mt]
        d.to_parquet(a.out / f"{pair}.parquet", index=False)
        print(f"{pair:10s} {script:10s} rows={len(d):6d} systems={d.system.nunique():3d} "
              f"segments={d.seg_idx.nunique():4d} roundtrip_ok={d.roundtrip_ok.mean():.4f} "
              f"no_native_script={(d.mt_native_chars == 0).sum()}", flush=True)


if __name__ == "__main__":
    main()

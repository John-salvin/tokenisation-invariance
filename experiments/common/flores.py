"""FLORES-200 tokeniser measures.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C
import intrinsic as I
import tokenisers as T

PIVOT = "eng_Latn"

INDIC_SCRIPT: list[str] = [
    "asm_Beng", "awa_Deva", "ben_Beng", "bho_Deva", "guj_Gujr", "hin_Deva",
    "hne_Deva", "kan_Knda", "kas_Deva", "mag_Deva", "mai_Deva", "mal_Mlym",
    "mar_Deva", "mni_Beng", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva",
    "sat_Olck", "sin_Sinh", "tam_Taml", "tel_Telu",
]

INDIC_ARABIC_SCRIPT: list[str] = ["urd_Arab", "snd_Arab", "kas_Arab"]

PAPER_FIVE: dict[str, str] = {
    "guj": "guj_Gujr", "tam": "tam_Taml", "mal": "mal_Mlym",
    "mar": "mar_Deva", "hin": "hin_Deva",
}

N_DEVTEST = 1012


class FloresError(RuntimeError):
    pass


def load_lang(code: str, split: str = "devtest") -> list[str]:
    path = C.FLORES_DIR / split / f"{code}.{split}"
    if not path.exists():
        raise FloresError(f"FLORES file missing: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != N_DEVTEST:
        raise FloresError(f"{code}: expected {N_DEVTEST} lines, got {len(lines)}")
    return lines


def compute(langs: list[str], tokenisers: list[str] | None = None,
            group: str = "indic_script") -> pd.DataFrame:
    tokenisers = tokenisers or T.PANEL
    eng = load_lang(PIVOT)
    rows = []
    for code in langs:
        tgt = load_lang(code)
        bp = [I.byte_premium(t, e) for t, e in zip(tgt, eng)]
        for tok in tokenisers:
            n_tok = [T.count(t, tok) for t in tgt]
            n_eng = [T.count(e, tok) for e in eng]
            n_wrd = [len(I.words(t)) for t in tgt]
            tp = [a / b for a, b in zip(n_tok, n_eng) if b > 0]
            rows.append({
                "corpus": "flores200_devtest",
                "group": group,
                "lang": code,
                "tokeniser": tok,
                "n_segments": len(tgt),
                "tp": sum(tp) / len(tp),
                "fertility": sum(n_tok) / sum(n_wrd),
                "cpt": sum(len(t) for t in tgt) / sum(n_tok),
                "wfr": sum(I.word_fragmentation_rate(t, tok) for t in tgt) / len(tgt),
                "byte_premium": sum(bp) / len(bp),
                "tokens_total": sum(n_tok),
                "words_total": sum(n_wrd),
            })
    return pd.DataFrame(rows)

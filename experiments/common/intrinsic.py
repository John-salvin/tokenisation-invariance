"""Intrinsic tokenisation measures (fertility, token parity, characters per token).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C
import tokenisers as T

MATTR_WINDOW = 50


def words(text: str) -> list[str]:
    return unicodedata.normalize("NFC", text).split()


def fertility(text: str, tok: str) -> float:
    w = words(text)
    if not w:
        return np.nan
    return T.count(text, tok) / len(w)


def chars_per_token(text: str, tok: str) -> float:
    n = T.count(text, tok)
    if n == 0:
        return np.nan
    return len(unicodedata.normalize("NFC", text)) / n


def word_fragmentation_rate(text: str, tok: str) -> float:
    w = words(text)
    if not w:
        return np.nan
    return float(np.mean([T.count(x, tok) > 1 for x in w]))


def token_parity(text_l: str, text_en: str, tok: str) -> float:
    n_en = T.count(text_en, tok)
    if n_en == 0:
        return np.nan
    return T.count(text_l, tok) / n_en


def byte_premium(text_l: str, text_en: str) -> float:
    n_en = len(unicodedata.normalize("NFC", text_en).encode("utf-8"))
    if n_en == 0:
        return np.nan
    return len(unicodedata.normalize("NFC", text_l).encode("utf-8")) / n_en


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float:
    n = len(tokens)
    if n == 0:
        return np.nan
    if n <= window:
        return len(set(tokens)) / n
    return float(np.mean([
        len(set(tokens[i:i + window])) / window for i in range(n - window + 1)
    ]))


def per_segment(seg: pd.DataFrame, tok: str) -> pd.DataFrame:
    out = seg[["segment_id", "lang", "condition"]].copy()
    out["tokeniser"] = tok
    out["fertility"] = [fertility(t, tok) for t in seg["target"]]
    out["cpt"] = [chars_per_token(t, tok) for t in seg["target"]]
    out["wfr"] = [word_fragmentation_rate(t, tok) for t in seg["target"]]
    out["tp"] = [token_parity(t, e, tok)
                 for t, e in zip(seg["target"], seg["source_en"])]
    out["byte_premium"] = [byte_premium(t, e)
                           for t, e in zip(seg["target"], seg["source_en"])]
    out["n_tokens"] = [T.count(t, tok) for t in seg["target"]]
    out["n_words"] = [len(words(t)) for t in seg["target"]]
    return out


def aggregate(per_seg: pd.DataFrame) -> pd.DataFrame:
    g = (per_seg
         .groupby(["tokeniser", "lang", "condition"], observed=True)
         .agg(fertility=("fertility", "mean"),
              cpt=("cpt", "mean"),
              wfr=("wfr", "mean"),
              tp=("tp", "mean"),
              byte_premium=("byte_premium", "mean"),
              n_tokens=("n_tokens", "sum"),
              n_words=("n_words", "sum"),
              n_segments=("segment_id", "count"))
         .reset_index())
    g["lang"] = pd.Categorical(g["lang"], categories=C.LANG_ORDER, ordered=True)
    return g.sort_values(["tokeniser", "lang", "condition"]).reset_index(drop=True)


def corpus_mattr(seg: pd.DataFrame, tok: str) -> pd.DataFrame:
    rows = []
    for (lang, cond), sub in seg.groupby(["lang", "condition"], observed=True):
        stream: list[str] = []
        tk = T.load(tok)
        for t in sub["target"]:
            stream.extend(tk.convert_ids_to_tokens(T.tokenise(t, tok)))
        rows.append({"tokeniser": tok, "lang": lang, "condition": cond,
                     "mattr": mattr(stream), "n_tokens": len(stream)})
    return pd.DataFrame(rows)

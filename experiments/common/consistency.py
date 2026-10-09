"""Tokenisation consistency rate (after Sun et al., 2023).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C
import tokenisers as T


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def word_token_ids(text: str, tok: str) -> tuple[list[str], list[tuple[int, ...]]]:
    text = _norm(text)
    tk = T.load(tok)
    words = text.split()
    if not words:
        return [], []

    spans, pos = [], 0
    for w in words:
        i = text.index(w, pos)
        spans.append((i, i + len(w)))
        pos = i + len(w)

    try:
        enc = tk(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
    except Exception as exc:
        raise NotImplementedError(
            f"tokeniser {tok!r} ({type(tk).__name__}) cannot return offset mappings, "
            f"so in-context token attribution is impossible: {exc!r}. "
            "TCR must not be computed for it -- a context-free fallback would report "
            "a meaningless TCR of 1.0."
        ) from exc

    per_word: list[tuple[int, ...]] = []
    for (a, b) in spans:
        got = [tid for tid, (s, e) in zip(ids, offsets)
               if e > s and s < b and e > a]
        per_word.append(tuple(got))
    return words, per_word


def tcr(hyp: str, ref: str, tok: str) -> dict:
    hw, hids = word_token_ids(hyp, tok)
    rw, rids = word_token_ids(ref, tok)
    if not hw or not rw:
        return {"tcr": np.nan, "n_shared": 0, "n_consistent": 0}

    matcher = SequenceMatcher(a=hw, b=rw, autojunk=False)
    n_shared = n_consistent = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            n_shared += 1
            if hids[block.a + k] == rids[block.b + k]:
                n_consistent += 1

    return {
        "tcr": n_consistent / n_shared if n_shared else np.nan,
        "n_shared": n_shared,
        "n_consistent": n_consistent,
    }


def per_segment(seg: pd.DataFrame, tok: str) -> pd.DataFrame:
    out = seg[["segment_id", "lang", "condition"]].copy()
    out["tokeniser"] = tok
    rec = [tcr(h, r, tok) for h, r in zip(seg["target"], seg["reference"])]
    for k in ("tcr", "n_shared", "n_consistent"):
        out[k] = [d[k] for d in rec]
    return out


def aggregate(per_seg: pd.DataFrame) -> pd.DataFrame:
    g = (per_seg.groupby(["tokeniser", "lang", "condition"], observed=True)
         .agg(tcr=("tcr", "mean"),
              n_shared=("n_shared", "sum"),
              n_segments=("segment_id", "count"))
         .reset_index())
    g["lang"] = pd.Categorical(g["lang"], categories=C.LANG_ORDER, ordered=True)
    return g.sort_values(["tokeniser", "lang", "condition"]).reset_index(drop=True)

"""Validity check for MorphScore's per-token decoding.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

MISMATCH_TOLERANCE = 0.01


def reconstructs(wordform: str, tokenizer, special_toks: set[str],
                  subword_prefix: str) -> bool:
    ids = tokenizer(wordform)["input_ids"]
    decoded = [tokenizer.decode(i) for i in ids]
    kept = [t for t in decoded if t not in special_toks]
    if subword_prefix:
        kept = [t.replace(subword_prefix, "") for t in kept]
    return "".join(kept) == wordform


def concat_mismatch_rate(wordforms: list[str], tokenizer, subword_prefix: str = "") -> dict:
    special = set(tokenizer.special_tokens_map.values())
    n = 0
    n_mismatch = 0
    for w in wordforms:
        if not isinstance(w, str) or not w:
            continue
        n += 1
        if not reconstructs(w, tokenizer, special, subword_prefix):
            n_mismatch += 1
    rate = (n_mismatch / n) if n else float("nan")
    return {"n_checked": n, "n_mismatch": n_mismatch, "mismatch_rate": rate,
            "valid": (rate <= MISMATCH_TOLERANCE) if n else False}


def gate_report(datasets: dict[str, pd.DataFrame], tokenizer, tok_name: str,
                 subword_prefix: str = "") -> pd.DataFrame:
    rows = []
    for lang, df in datasets.items():
        words = df["wordform"].astype(str).unique().tolist()
        r = concat_mismatch_rate(words, tokenizer, subword_prefix)
        r["tokeniser"] = tok_name
        r["lang"] = lang
        rows.append(r)
    return pd.DataFrame(rows)

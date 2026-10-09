"""The tokeniser panel, loaded from staged snapshots.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constants as C

PANEL: list[str] = [
    "xlmr", "mbert", "byt5", "gpt2",
    "mt5", "rembert", "nllb", "indicbert", "llama31", "qwen3", "gemma2",
]

ORIGINAL_FOUR: dict[str, str] = {
    "xlmr": "xlmr", "mbert": "mbert", "byt5": "byt5", "gpt2": "gpt2",
}

MIRRORED: dict[str, str] = {
    "llama31": "meta-llama/Llama-3.1-8B",
    "gemma2": "google/gemma-2-9b",
}


class TokeniserError(RuntimeError):
    pass


@functools.lru_cache(maxsize=None)
def load(name: str):
    from transformers import AutoTokenizer

    if name not in PANEL:
        raise TokeniserError(f"unknown tokeniser {name!r}; panel is {PANEL}")
    path = C.TOKENISER_DIR / name
    if not path.exists():
        raise TokeniserError(f"tokeniser {name!r} not staged at {path}")
    try:
        return AutoTokenizer.from_pretrained(str(path), use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(str(path), use_fast=False)


def tokenise(text: str, name: str, *, add_special_tokens: bool = False) -> list[int]:
    if not isinstance(text, str):
        raise TokeniserError(f"expected str, got {type(text).__name__}")
    return load(name)(text, add_special_tokens=add_special_tokens)["input_ids"]


def count(text: str, name: str) -> int:
    return len(tokenise(text, name))


def verify_against_workbook(seg: pd.DataFrame, *, tol_rows: int = 0) -> pd.DataFrame:
    rows = []
    for name in ORIGINAL_FOUR:
        col = f"tok_count_{name}"
        if col not in seg.columns:
            raise TokeniserError(f"workbook column {col!r} missing from segments frame")
        for cond in C.CONDITIONS:
            sub = seg[seg.condition == cond]
            recomputed = sub["target"].map(lambda t: count(t, name))
            committed = sub[col].astype("int64")
            diff = (recomputed.values != committed.values)
            n_bad = int(diff.sum())
            rows.append({
                "tokeniser": name, "condition": cond, "n": len(sub),
                "n_mismatch": n_bad,
                "pct_mismatch": 100.0 * n_bad / len(sub),
                "mean_committed": float(committed.mean()),
                "mean_recomputed": float(recomputed.mean()),
            })
    rep = pd.DataFrame(rows)
    bad = rep[rep.n_mismatch > tol_rows]
    if len(bad):
        raise TokeniserError(
            "tokeniser counts do not reproduce the committed workbook values:\n"
            + bad.to_string(index=False)
            + "\n\nDo NOT proceed to the new tokenisers until this is understood."
        )
    return rep


def panel_manifest() -> dict:
    man = {}
    for name in PANEL:
        try:
            t = load(name)
        except Exception as exc:
            man[name] = {"status": "FAILED", "error": repr(exc)}
            continue
        man[name] = {
            "status": "ok",
            "class": type(t).__name__,
            "vocab_size": int(t.vocab_size),
            "local_dir": str(C.TOKENISER_DIR / name),
            "mirror_of": MIRRORED.get(name),
        }
    return man


if __name__ == "__main__":
    print(json.dumps(panel_manifest(), indent=2))

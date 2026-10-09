#!/usr/bin/env python3
"""Regenerate the two WMT tokeniser statistics of Section 4 (CPU, ~2 min).

  wmt_roundtrip.csv     how many of the first 60 lines of one system output
                        survive decode(tokenise(x)) == x under XLM-R, per pair.
                        Chinese fails on every line because XLM-R's normaliser
                        rewrites fullwidth punctuation.
  wmt_token_parity.csv  token parity |t(target)| / |t(source)| for WMT25
                        Italian and WMT24 German, on the first 120 segments
                        and on every scored segment (`full`). Kept for the
                        record; the paper's effect-size analysis uses the
                        covariates of experiments/tokenisation/effect_covariates.py.

Needs the public mt-metrics-eval data (WMT24, WMT25), see
https://github.com/google-research/mt-metrics-eval, and the WMT24 MQM
workbook converted in data/wmt/.

    python experiments/tokenisation/wmt_tokeniser_stats.py --mtme ~/.mt-metrics-eval/mt-metrics-eval-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tokinv import sampler as S  # noqa: E402
from tokinv import tokenisers as T  # noqa: E402

OUT = ROOT / "data" / "derived" / "tokeniser_panel"
ROUNDTRIP_PAIRS = [("zh", "en-zh_CN"), ("ja", "en-ja_JP"), ("ko", "en-ko_KR"), ("cs", "en-cs_CZ"),
                   ("ru", "en-ru_RU"), ("uk", "en-uk_UA"), ("bho", "en-bho_IN"), ("ar", "en-ar_EG")]
N_ROUNDTRIP, N_PARITY = 60, 120


def lines(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def roundtrip(mtme: Path) -> pd.DataFrame:
    tok = T.get("xlmr")
    norm = lambda x: tok.decode(tok(x, add_special_tokens=False)["input_ids"])
    rows = []
    for lang, lp in ROUNDTRIP_PAIRS:
        first = sorted((mtme / "wmt25" / "system-outputs" / lp).glob("*.txt"))[0]
        xs = [S.normalize_text_for_reseg(l) for l in lines(first)[:N_ROUNDTRIP] if l.strip()]
        rows.append({"lang": lang, "pair": lp, "system_file": first.name, "n_lines": len(xs),
                     "roundtrip_ok_of_60": sum(norm(x) == x for x in xs)})
    return pd.DataFrame(rows)


def _scored_systems(mtme: Path, lp: str) -> tuple[list[str], dict]:
    by = {}
    f = mtme / "wmt25" / "human-scores" / f"{lp}.esa-merged.seg.score"
    for line in open(f, encoding="utf-8"):
        s, v = line.rstrip("\n").split("\t")
        by.setdefault(s, []).append(v)
    scored = sorted(s for s, v in by.items()
                    if any(x != "None" for x in v) and not s.lower().startswith("ref"))
    return scored, by


def parity_italian(mtme: Path) -> list[dict]:
    tok = T.get("xlmr")
    lp = "en-it_IT"
    d = mtme / "wmt25"
    src = lines(d / "sources" / f"{lp}.txt")
    scored, by = _scored_systems(mtme, lp)
    tgt = lines(d / "system-outputs" / lp / f"{scored[0]}.txt")
    # as printed: the first 120 non-empty lines of the first scored system
    t = [x for x in tgt[:N_PARITY] if x.strip()]
    s = [x for x in src[:N_PARITY] if x.strip()]
    sample = np.mean([len(tok.tokenize(a)) / len(tok.tokenize(b))
                      for a, b in zip(t, s) if len(tok.tokenize(b))])
    # full: every scored (system, segment) pair
    full = []
    for sysname in scored:
        out = lines(d / "system-outputs" / lp / f"{sysname}.txt")
        for i, (a, v) in enumerate(zip(out, by[sysname])):
            if v != "None" and a.strip() and src[i].strip():
                full.append(len(tok.tokenize(a)) / len(tok.tokenize(src[i])))
    return [{"setting": "it_IT", "token_parity": round(float(sample), 3), "basis": f"first{N_PARITY}"},
            {"setting": "it_IT", "token_parity": round(float(np.mean(full)), 3), "basis": "full"}]


def parity_german() -> list[dict]:
    tok = T.get("xlmr")
    w = pd.read_parquet(ROOT / "data" / "wmt" / "wmt24_mqm_en-de_en-es.parquet")
    d = (w[w.pair == "deu"].dropna(subset=["source", "target", "refA", "mqm_score"])
           .drop_duplicates(subset=["system", "seg_id"]))
    tp = lambda g: float(np.mean([len(tok.tokenize(str(a))) / len(tok.tokenize(str(b)))
                                  for a, b in zip(g.target, g.source) if len(tok.tokenize(str(b)))]))
    return [{"setting": "deu", "token_parity": round(tp(d.head(N_PARITY)), 3), "basis": f"first{N_PARITY}"},
            {"setting": "deu", "token_parity": round(tp(d), 3), "basis": "full"}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtme", type=Path, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rt = roundtrip(a.mtme)
    rt.to_csv(OUT / "wmt_roundtrip.csv", index=False)
    print(rt.to_string(index=False))
    tp = pd.DataFrame(parity_italian(a.mtme) + parity_german())
    tp.to_csv(OUT / "wmt_token_parity.csv", index=False)
    print(tp.to_string(index=False))


if __name__ == "__main__":
    main()

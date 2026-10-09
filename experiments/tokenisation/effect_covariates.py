#!/usr/bin/env python3
"""Per-segment tokeniser covariates for the Table 1 settings (App.
Predictors of Effect Size): source and target XLM-R token counts, target word
count, and fragmentation headroom (the most XLM-R tokens any segmentation of
the target can have, over its canonical count). Computed once and written to
data/derived/effect_predictors/segment_covariates.parquet, so the analysis
(tokinv.effect_predictors) needs no raw WMT files.

    python experiments/tokenisation/effect_covariates.py --mtme ~/.mt-metrics-eval/mt-metrics-eval-v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tokinv import resegmentation, sampler as S, tokenisers as T  # noqa: E402
from tokinv.paths import DATA, INDIC  # noqa: E402

OUT = DATA / "derived" / "effect_predictors" / "segment_covariates.parquet"


def headroom(text: str, vocab: set, maxp: int, tok) -> float:
    """Largest token count of any segmentation of `text` from the vocabulary,
    over the canonical count (a longest-path dynamic programme over the vocabulary)."""
    s = "▁" + text.replace(" ", "▁")
    canon = len(tok.tokenize(text)) or 1
    mx = [-1] * (len(s) + 1)
    mx[len(s)] = 0
    for i in range(len(s) - 1, -1, -1):
        best = -1
        for j in range(i + 1, min(len(s), i + maxp) + 1):
            if mx[j] != -1 and s[i:j] in vocab:
                best = max(best, 1 + mx[j])
        mx[i] = best
    return mx[0] / canon if mx[0] > 0 else float("nan")


def segments(mtme: Path) -> pd.DataFrame:
    """(setting, key, source, target) for the canonical rows of every Table 1 run."""
    rows = []
    seg = pd.read_parquet(INDIC / "segments.parquet")
    nat = seg[seg.condition == "native"].set_index("segment_id")
    lat = pd.read_parquet(DATA / "wmt" / "wmt24_mqm_en-de_en-es.parquet") \
        .drop_duplicates(subset=["pair", "system", "seg_id"]).set_index(["pair", "system", "seg_id"])
    for corpus, labels, setting, file, subset in resegmentation.SETTINGS:
        # CJK settings are included for the vocabulary appendix (fragmentation
        # ceiling by script); the effect-size analysis uses Table 1 settings only
        d = resegmentation.load_setting(file, subset)
        if file == "indic_comet22":
            for u in sorted(d.unit.unique()):
                rows.append({"setting": setting, "key": str(u), "source": nat.loc[u, "source_en"],
                             "target": nat.loc[u, "target"]})
        elif file == "latin_wmt24_comet22":
            c = d[d.bucket == "canonical"][["system", "unit"]].drop_duplicates()
            for sysn, u in c.itertuples(index=False):
                r = lat.loc[(subset, sysn, u)]
                rows.append({"setting": setting, "key": f"{sysn}|{u}", "source": r["source"], "target": r["target"]})
        else:
            pair = file[len("wmt_"):].split("_hp")[0].split("_pw")[0].split("_kiwi22")[0].split("_cjk")[0]
            ts = "wmt25" if "_" in pair else "wmt24"
            src = [l.rstrip("\n") for l in open(mtme / ts / "sources" / f"{pair}.txt", encoding="utf-8")]
            c = d[d.bucket == "canonical"][["system", "unit"]].drop_duplicates()
            cache = {}
            for sysn, u in c.itertuples(index=False):
                if sysn not in cache:
                    cache[sysn] = [l.rstrip("\n") for l in
                                   open(mtme / ts / "system-outputs" / pair / f"{sysn}.txt", encoding="utf-8")]
                rows.append({"setting": setting, "key": f"{sysn}|{int(u)}", "source": src[int(u)],
                             "target": cache[sysn][int(u)]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtme", type=Path, required=True)
    a = ap.parse_args()
    tok = T.get("xlmr")
    vocab = set(tok.get_vocab())
    maxp = max(len(p) for p in vocab) + 1
    d = segments(a.mtme)
    tgt = [S.normalize_text_for_reseg(str(t)) for t in d.target]
    src = [S.normalize_text_for_reseg(str(s)) for s in d.source]
    d["n_tgt_tok"] = [len(tok.tokenize(t)) for t in tgt]
    d["n_src_tok"] = [len(tok.tokenize(s)) for s in src]
    d["n_tgt_words"] = [len(t.split()) for t in tgt]
    d["headroom"] = [headroom(t, vocab, maxp, tok) for t in tgt]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["source", "target"]).to_parquet(OUT, index=False)
    print(d.groupby("setting").size().to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main()

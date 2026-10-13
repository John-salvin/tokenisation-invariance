"""Build LoRA training data for WMT pairs in the format of lora_lolo_pipeline.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(LIVE / "src"))
import noncanonical as NC
import tokenisers as T

DATA_ROOT = Path(os.environ.get("MTME_ROOT", SB_ROOT / "staging" / "mtme" / "mt-metrics-eval-v2"))
OUT = ROB / "data" / "interim"
TOKENISER = "xlmr"
MAX_SEQ_IDS, MAX_CANON_K = 510, 160
SEED = 20260930
MAIN_BUCKETS = ["[1.00,1.25)", "[1.25,1.50)", "[1.50,1.75)", "[1.75,2.00)",
                "[2.00,2.25)", "[2.25,2.50)", "[2.50,2.75)", "[2.75,3.00)"]
SCORE_PREF = ["mqm", "esa-merged", "esa"]


def read_lines(p: Path) -> list[str]:
    return [l.rstrip("\n") for l in open(p, encoding="utf-8")]


def score_file(d: Path, lp: str):
    for k in SCORE_PREF:
        f = d / "human-scores" / f"{lp}.{k}.seg.score"
        if f.exists():
            return f, k
    g = sorted((d / "human-scores").glob(f"{lp}.*.seg.score"))
    if not g:
        raise SystemExit(f"no human scores for {lp}")
    return g[0], g[0].name


def build_pair(testset: str, lp: str, n_seg: int, lang_tag: str) -> pd.DataFrame:
    d = DATA_ROOT / testset
    src = read_lines(d / "sources" / f"{lp}.txt")
    ref = read_lines(d / "references" / f"{lp}.refA.txt")
    sf, kind = score_file(d, lp)
    by: dict[str, list[str]] = {}
    for line in open(sf, encoding="utf-8"):
        s, v = line.rstrip("\n").split("\t")
        by.setdefault(s, []).append(v)
    systems = [s for s, v in by.items()
               if any(x != "None" for x in v) and not s.lower().startswith("ref")]
    print(f"  {lp}: {len(systems)} systems, labels={kind}", flush=True)

    recs = []
    for s in sorted(systems):
        out = read_lines(d / "system-outputs" / lp / f"{s}.txt")
        for i, (txt, sc) in enumerate(zip(out, by[s])):
            if sc == "None":
                continue
            recs.append({"segment_id": f"{lang_tag}_{s}_{i}", "lang": lang_tag,
                         "source": src[i], "refA": ref[i], "target": txt,
                         "mqm_score": float(sc)})
    df = pd.DataFrame(recs)

    keep = []
    for _, r in df.iterrows():
        try:
            ck = len(T.tokenise(NC.normalize_text_for_reseg(str(r.target)), TOKENISER,
                                add_special_tokens=False))
            sk = len(T.tokenise(str(r.source), TOKENISER, add_special_tokens=False))
            rk = len(T.tokenise(str(r.refA), TOKENISER, add_special_tokens=False))
        except Exception:
            keep.append(False); continue
        keep.append(ck <= MAX_CANON_K and sk <= MAX_SEQ_IDS and rk <= MAX_SEQ_IDS)
    df = df[pd.Series(keep, index=df.index)]
    print(f"  {lp}: {len(df)} rows after the length filter", flush=True)
    pool = df.sample(n=min(n_seg, len(df)), random_state=SEED).reset_index(drop=True)

    rows, n_skip = [], 0
    for _, r in pool.iterrows():
        text = NC.normalize_text_for_reseg(str(r.target))
        if not text.strip():
            continue
        try:
            src_ids = T.tokenise(str(r.source), TOKENISER, add_special_tokens=False)
            ref_ids = T.tokenise(str(r.refA), TOKENISER, add_special_tokens=False)
            canon = T.tokenise(text, TOKENISER, add_special_tokens=False)
            NC.assert_round_trip(text, canon, TOKENISER)
        except Exception:
            n_skip += 1; continue
        base = {"segment_id": r.segment_id, "lang": r.lang,
                "src_ids": src_ids, "ref_ids": ref_ids, "mqm_score": r.mqm_score}
        rows.append({**base, "view": "canonical", "mt_ids": canon})
        for b, (lo, hi) in zip(MAIN_BUCKETS, NC.RATIO_BUCKETS):
            try:
                s = NC.sample_bucket(text, TOKENISER, lo, hi, random.Random(0))
                if not s["reachable"]:
                    continue
                NC.assert_round_trip(text, s["ids"], TOKENISER)
                if len(s["ids"]) > MAX_SEQ_IDS:
                    continue
                rows.append({**base, "view": f"noncanon_{b}", "mt_ids": s["ids"]})
            except NC.NoncanonicalError:
                pass
    print(f"  {lp}: {len(rows)} training views, {n_skip} segments skipped", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="testset:pair[:tag], e.g. wmt25:en-ru_RU:rus")
    ap.add_argument("--n-seg", type=int, default=400)
    ap.add_argument("--out", default=str(OUT / "lora_train_wmt.parquet"))
    a = ap.parse_args()
    frames = []
    for spec in a.pairs:
        parts = spec.split(":")
        ts, lp = parts[0], parts[1]
        tag = parts[2] if len(parts) > 2 else lp.split("-")[-1].split("_")[0]
        print(f"== {ts} {lp} -> lang tag {tag!r}", flush=True)
        frames.append(build_pair(ts, lp, a.n_seg, tag))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(a.out)
    print(f"\nwrote {a.out}: {len(df)} rows, langs {sorted(df.lang.unique())}")
    print(df.groupby(['lang','view']).size().groupby('lang').sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

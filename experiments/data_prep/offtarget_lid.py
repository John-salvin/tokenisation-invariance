#!/usr/bin/env python3
"""Off-target systems by language identification.

A system is off-target in a setting if fewer than half of its rated outputs
get the target variety's label as GlotLID v3's top-1 prediction. The criterion
reads only the output text; the share of zero human scores is computed
afterwards and only compared with it. References are identified too: a
setting whose references get the target label less than 90 percent of the
time is reported as unreliable and drops no system.

Needs `fasttext-numpy2-wheel` and `huggingface_hub` (internet, once, for the
model); kept out of the main environment, whose numpy is pinned below 2.

    python experiments/data_prep/offtarget_lid.py --mtme ~/.mt-metrics-eval/mt-metrics-eval-v2 \
        --out data/derived/offtarget
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

MODEL_SHA256 = "a818b6bd42a628ab47d3dfc1578c7ea615c45381f3494c42535e31e8c4cafc9e"
MAJORITY = 0.5
REF_MIN = 0.9
# setting: (test set, human-score file kind, target label)
SETTINGS = {
    "en-cs": ("wmt24", "esa", "ces_Latn"), "en-uk": ("wmt24", "esa", "ukr_Cyrl"),
    "en-ru": ("wmt24", "esa", "rus_Cyrl"), "en-is": ("wmt24", "esa", "isl_Latn"),
    "en-hi": ("wmt24", "esa", "hin_Deva"),
    "en-cs_CZ": ("wmt25", "esa-merged", "ces_Latn"), "en-is_IS": ("wmt25", "esa-merged", "isl_Latn"),
    "en-bho_IN": ("wmt25", "esa-merged", "bho_Deva"), "en-uk_UA": ("wmt25", "esa-merged", "ukr_Cyrl"),
    "en-ar_EG": ("wmt25", "esa-merged", "arz_Arab"), "en-ru_RU": ("wmt25", "esa-merged", "rus_Cyrl"),
    "en-et_EE": ("wmt25", "esa-merged", "ekk_Latn"), "en-it_IT": ("wmt25", "esa-merged", "ita_Latn"),
    "en-ja_JP": ("wmt25", "esa-merged", "jpn_Jpan"), "en-zh_CN": ("wmt25", "esa-merged", "cmn_Hani"),
    "en-ko_KR": ("wmt25", "mqm", "kor_Hang"), "en-sr_Cyrl_RS": ("wmt25", "esa-merged", "srp_Cyrl"),
}


def clean(s: str) -> str:
    return " ".join(s.replace("\\n", " ").split())


def rated_rows(mtme: Path, pair: str, ts: str, kind: str) -> pd.DataFrame:
    d = mtme / ts
    by: dict[str, list[str]] = {}
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
                rows.append({"system": s, "seg_idx": i, "human": float(v), "text": mt})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtme", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import fasttext
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("cis-lmu/glotlid", "model_v3.bin")
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == MODEL_SHA256, "GlotLID model changed"
    model = fasttext.load_model(path)
    top = lambda texts: [l[0].replace("__label__", "") for l in model.predict([clean(t) for t in texts], k=1)[0]]

    a.out.mkdir(parents=True, exist_ok=True)
    segs, summary = [], []
    for pair, (ts, kind, target) in SETTINGS.items():
        d = rated_rows(a.mtme, pair, ts, kind)
        d["label"] = top(d.text)
        refs = [l.rstrip("\n") for l in open(a.mtme / ts / "references" / f"{pair}.refA.txt", encoding="utf-8")]
        refs = [r for r in refs if r.strip()]
        ref_share = sum(l == target for l in top(refs)) / len(refs)
        reliable = ref_share >= REF_MIN
        for sys_, g in d.groupby("system"):
            share = float((g.label == target).mean())
            summary.append({"setting": pair, "target": target, "ref_target_share": round(ref_share, 4),
                            "lid_reliable": reliable, "system": sys_, "n": len(g),
                            "target_share": round(share, 4),
                            "top_other_label": g.label[g.label != target].mode().iat[0] if share < 1 else "",
                            "off_target": bool(reliable and share < MAJORITY),
                            "zero_share": round(float((g.human == 0).mean()), 4),
                            "zero_rule": bool((g.human == 0).mean() > MAJORITY)})
        segs.append(d.drop(columns=["text", "human"]).assign(setting=pair))
    s = pd.DataFrame(summary)
    s.to_csv(a.out / "offtarget_systems.csv", index=False)
    pd.concat(segs).to_parquet(a.out / "lid_segments.parquet", index=False)
    for pair, g in s.groupby("setting", sort=False):
        print(f"{pair:14s} ref={g.ref_target_share.iat[0]:.3f} reliable={g.lid_reliable.iat[0]} "
              f"lid_off={sorted(g.system[g.off_target])} zero_rule={sorted(g.system[g.zero_rule])}")


if __name__ == "__main__":
    main()

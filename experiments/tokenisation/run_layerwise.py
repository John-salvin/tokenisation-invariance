#!/usr/bin/env python3
"""Layerwise retrieval probe of COMET-22's encoder.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import layerwise as LW
import provenance as PROV
import tokenisers as T

TOKENISER = "xlmr"
NONCANONICAL_BUCKET = "[1.50,1.75)"
N_PER_LANG = 200
BATCH_SIZE = 16


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    C.OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    reseg_samples_path = C.DATA_INTERIM / "reseg_samples.parquet"
    if not reseg_samples_path.exists():
        raise RuntimeError(f"{reseg_samples_path} missing -- needs the earlier re-segmentation sampling output")
    reseg_samples = pd.read_parquet(reseg_samples_path)
    nc = reseg_samples[(reseg_samples.direction == "forward") & (reseg_samples.bucket == NONCANONICAL_BUCKET)
                    & (reseg_samples.seed == 0) & reseg_samples.mt_ids.notna()]

    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid]
    nat = seg[seg.condition == "native"].set_index(["lang", "segment_id"])
    rom = seg[seg.condition == "romanised"].set_index(["lang", "segment_id"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    model, tokenizer = LW.load_xlmr_encoder(device=device)

    rows = []
    for lang in C.LANG_ORDER:
        lang_nc = nc[nc.lang == lang].reset_index(drop=True)
        if len(lang_nc) > N_PER_LANG:
            lang_nc = lang_nc.sample(n=N_PER_LANG, random_state=C.SEED)
        seg_ids = lang_nc["segment_id"].tolist()
        print(f"{lang}: {len(seg_ids)} segments with a reachable {NONCANONICAL_BUCKET} sample")
        if len(seg_ids) < 20:
            print(f"  too few segments for a meaningful retrieval pool, skipping {lang}")
            continue

        en_ids = [T.tokenise(str(nat.loc[(lang, sid), "source_en"]), TOKENISER,
                             add_special_tokens=False) for sid in seg_ids]
        nat_ids = [T.tokenise(str(nat.loc[(lang, sid), "target"]), TOKENISER,
                              add_special_tokens=False) for sid in seg_ids]
        rom_ids = [T.tokenise(str(rom.loc[(lang, sid), "target"]), TOKENISER,
                              add_special_tokens=False) for sid in seg_ids]
        nc_ids = [list(x) for x in lang_nc["mt_ids"].tolist()]

        en_emb = LW.embed_batch(en_ids, model, tokenizer, device, batch_size=BATCH_SIZE)
        nat_emb = LW.embed_batch(nat_ids, model, tokenizer, device, batch_size=BATCH_SIZE)
        rom_emb = LW.embed_batch(rom_ids, model, tokenizer, device, batch_size=BATCH_SIZE)
        nc_emb = LW.embed_batch(nc_ids, model, tokenizer, device, batch_size=BATCH_SIZE)

        n_layers = en_emb.shape[1]
        for layer in range(n_layers):
            for cond_name, tgt_emb in [("native", nat_emb), ("romanised", rom_emb),
                                       ("noncanonical", nc_emb)]:
                acc = LW.retrieval_acc1(en_emb[:, layer, :], tgt_emb[:, layer, :])
                rows.append({"lang": lang, "condition": cond_name, "layer": layer,
                            "acc1": acc, "n": len(seg_ids)})
        print(f"  {lang} done, {len(seg_ids)} segments x {n_layers} layers x 3 conditions")

    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_TABLES / "layerwise_acc1.csv", index=False)
    print(df.groupby(["lang", "condition"])["acc1"].max())

    langs_present = [l for l in C.LANG_ORDER if l in df.lang.unique()]
    fig, axes = plt.subplots(1, len(langs_present), figsize=(4 * len(langs_present), 4),
                             sharey=True)
    if len(langs_present) == 1:
        axes = [axes]
    colors = {"native": "tab:blue", "romanised": "tab:orange", "noncanonical": "tab:green"}
    for ax, lang in zip(axes, langs_present):
        sub = df[df.lang == lang]
        for cond in ["native", "romanised", "noncanonical"]:
            csub = sub[sub.condition == cond].sort_values("layer")
            ax.plot(csub["layer"], csub["acc1"], label=cond, color=colors[cond], marker="o",
                   markersize=2)
        ax.set_title(lang.upper())
        ax.set_xlabel("layer")
    axes[0].set_ylabel("EN<->target retrieval acc@1")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("layerwise probe: cross-lingual retrieval accuracy by layer, XLM-R-large")
    fig.tight_layout()
    fig.savefig(C.OUT_FIGURES / "layerwise.png", dpi=150)
    print(f"\nFigure written to {C.OUT_FIGURES / 'layerwise.png'}")

    peak_rows = []
    for lang in langs_present:
        for cond in ["native", "romanised", "noncanonical"]:
            sub = df[(df.lang == lang) & (df.condition == cond)]
            peak = sub.loc[sub.acc1.idxmax()]
            peak_rows.append({"lang": lang, "condition": cond,
                             "peak_layer": int(peak.layer), "peak_acc1": float(peak.acc1)})
    peak_df = pd.DataFrame(peak_rows)
    print("\nPeak layer per (lang, condition):")
    print(peak_df.to_string(index=False))

    PROV.register(
        "layerwise", {
            "acc1_by_layer": df.to_dict("records"),
            "peak_layer": peak_df.to_dict("records"),
            "noncanonical_bucket_used": NONCANONICAL_BUCKET,
            "n_per_lang_target": N_PER_LANG,
            "figure": "results/figures/layerwise.png",
        },
        PROV.stamp(Path(__file__), modules=["constants", "layerwise", "tokenisers"], seed=C.SEED,
                   note="layerwise probe: layerwise cross-lingual retrieval"))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

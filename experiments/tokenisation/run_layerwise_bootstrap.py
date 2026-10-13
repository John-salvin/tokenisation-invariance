#!/usr/bin/env python3
"""Layerwise retrieval probe over several seeds and segment subsamples.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ORIG_REPO = Path.home() / "tokinv_work/repo/panel-pipeline"
NEW_REPO = Path.home() / "tokinv_work/repo/cluster_copy"
sys.path.insert(0, str(ORIG_REPO / "src"))
import constants as C
import layerwise as LW
import tokenisers as T
import stats as ST

OUT_TABLES = NEW_REPO / "results/tables"
OUT_FIGURES = NEW_REPO / "results/figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

TOKENISER = "xlmr"
MAIN_BUCKET = "[1.50,1.75)"
ALT_BUCKETS = ["[1.25,1.50)", "[1.75,2.00)"]
SEEDS = [0, 1, 2]
N_PER_LANG = 200
BATCH_SIZE = 16
N_BOOT = 2000


def acc1_and_correct(en_emb, tgt_emb):
    en_n = en_emb / (np.linalg.norm(en_emb, axis=1, keepdims=True) + 1e-9)
    tgt_n = tgt_emb / (np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-9)
    sims = en_n @ tgt_n.T
    preds = sims.argmax(axis=1)
    correct = (preds == np.arange(len(preds))).astype(float)
    return float(correct.mean()), correct


def boot_ci(correct: np.ndarray) -> tuple[float, float]:
    d = correct.reshape(-1, 1)
    out = ST.bca_bootstrap(d, lambda m: m.mean(), n_boot=N_BOOT, alpha=0.05, seed=42)
    return out["ci_low"], out["ci_high"]


def main():
    reseg_samples_path = C.DATA_INTERIM / "reseg_samples.parquet"
    reseg_samples = pd.read_parquet(reseg_samples_path)

    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid]
    nat = seg[seg.condition == "native"].set_index(["lang", "segment_id"])
    rom = seg[seg.condition == "romanised"].set_index(["lang", "segment_id"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    model, tokenizer = LW.load_xlmr_encoder(device=device)

    rows = []
    seed0_cache = {}

    def embed_condition(seg_ids, texts):
        ids = [T.tokenise(str(t), TOKENISER, add_special_tokens=False) for t in texts]
        return LW.embed_batch(ids, model, tokenizer, device, batch_size=BATCH_SIZE)

    print("=" * 78); print("PART 1: 3 seeds x 5 langs, native/romanised/noncanonical"); print("=" * 78, flush=True)
    for seed in SEEDS:
        for lang in C.LANG_ORDER:
            nc = reseg_samples[(reseg_samples.direction == "forward") & (reseg_samples.bucket == MAIN_BUCKET)
                            & (reseg_samples.seed == seed) & reseg_samples.mt_ids.notna()
                            & (reseg_samples.lang == lang)].reset_index(drop=True)
            if len(nc) > N_PER_LANG:
                nc = nc.sample(n=N_PER_LANG, random_state=1000 + seed)
            seg_ids = nc["segment_id"].tolist()
            print(f"seed={seed} {lang}: n={len(seg_ids)}", flush=True)
            if len(seg_ids) < 20:
                print("  too few, skipping"); continue

            en_texts = [str(nat.loc[(lang, sid), "source_en"]) for sid in seg_ids]
            nat_texts = [str(nat.loc[(lang, sid), "target"]) for sid in seg_ids]
            rom_texts = [str(rom.loc[(lang, sid), "target"]) for sid in seg_ids]
            nc_ids_list = [list(x) for x in nc["mt_ids"].tolist()]

            en_emb = embed_condition(seg_ids, en_texts)
            nat_emb = embed_condition(seg_ids, nat_texts)
            rom_emb = embed_condition(seg_ids, rom_texts)
            nc_emb = LW.embed_batch(nc_ids_list, model, tokenizer, device, batch_size=BATCH_SIZE)

            if seed == 0:
                seed0_cache[lang] = {
                    "seg_ids": seg_ids, "en_emb": en_emb, "nat_emb": nat_emb, "rom_emb": rom_emb,
                }

            n_layers = en_emb.shape[1]
            for layer in range(n_layers):
                for cond_name, tgt_emb in [("native", nat_emb), ("romanised", rom_emb),
                                           ("noncanonical", nc_emb)]:
                    acc, correct = acc1_and_correct(en_emb[:, layer, :], tgt_emb[:, layer, :])
                    lo, hi = boot_ci(correct)
                    rows.append({"lang": lang, "condition": cond_name, "layer": layer,
                                "replicate_type": "seed", "replicate_id": str(seed),
                                "n": len(seg_ids), "acc1": acc,
                                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi})
            print(f"  seed={seed} {lang} done", flush=True)

    print("\n" + "=" * 78); print("PART 2: bucket-choice robustness (seed=0 only)"); print("=" * 78, flush=True)
    for lang in C.LANG_ORDER:
        cache = seed0_cache.get(lang)
        if cache is None:
            print(f"  {lang}: no seed=0 cache, skipping bucket check"); continue
        base_seg_ids = cache["seg_ids"]
        base_index = {sid: i for i, sid in enumerate(base_seg_ids)}
        for alt_bucket in ALT_BUCKETS:
            alt = reseg_samples[(reseg_samples.direction == "forward") & (reseg_samples.bucket == alt_bucket)
                             & (reseg_samples.seed == 0) & reseg_samples.mt_ids.notna()
                             & (reseg_samples.lang == lang)]
            common_ids = [sid for sid in alt["segment_id"].tolist() if sid in base_index]
            print(f"  {lang} {alt_bucket}: {len(common_ids)} segments in common with the "
                  f"main-bucket seed=0 pool (of {len(alt)} available in this bucket)", flush=True)
            if len(common_ids) < 20:
                print("    too few overlapping segments for a meaningful pool, skipping")
                continue
            idx = [base_index[sid] for sid in common_ids]
            en_emb = cache["en_emb"][idx]
            nat_emb = cache["nat_emb"][idx]
            rom_emb = cache["rom_emb"][idx]
            alt_sub = alt.set_index("segment_id").loc[common_ids]
            nc_ids_list = [list(x) for x in alt_sub["mt_ids"].tolist()]
            nc_emb = LW.embed_batch(nc_ids_list, model, tokenizer, device, batch_size=BATCH_SIZE)

            n_layers = en_emb.shape[1]
            for layer in range(n_layers):
                for cond_name, tgt_emb in [("native", nat_emb), ("romanised", rom_emb),
                                           ("noncanonical", nc_emb)]:
                    acc, correct = acc1_and_correct(en_emb[:, layer, :], tgt_emb[:, layer, :])
                    lo, hi = boot_ci(correct)
                    rows.append({"lang": lang, "condition": cond_name, "layer": layer,
                                "replicate_type": "bucket", "replicate_id": alt_bucket,
                                "n": len(common_ids), "acc1": acc,
                                "bootstrap_ci_low": lo, "bootstrap_ci_high": hi})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLES / "layerwise_bootstrap.csv", index=False)
    print(f"\nWrote {OUT_TABLES / 'layerwise_bootstrap.csv'} ({len(df)} rows)")

    seed_rows = df[df.replicate_type == "seed"]
    summ = (seed_rows.groupby(["lang", "condition", "layer"])["acc1"]
           .agg(["mean", "std", "min", "max", "count"]).reset_index()
           .rename(columns={"mean": "seed_mean_acc1", "std": "seed_sd_acc1",
                            "min": "seed_min_acc1", "max": "seed_max_acc1", "count": "n_seeds"}))
    summ.to_csv(OUT_TABLES / "layerwise_seed_summary.csv", index=False)
    print(f"Wrote {OUT_TABLES / 'layerwise_seed_summary.csv'} ({len(summ)} rows)")

    peak_rows = []
    for (lang, cond), g in summ.groupby(["lang", "condition"]):
        peak = g.loc[g.seed_mean_acc1.idxmax()]
        peak_rows.append({"lang": lang, "condition": cond, "peak_layer": int(peak.layer),
                          "peak_seed_mean_acc1": float(peak.seed_mean_acc1),
                          "peak_seed_sd_acc1": float(peak.seed_sd_acc1)})
    peak_df = pd.DataFrame(peak_rows)
    peak_df.to_csv(OUT_TABLES / "layerwise_peak_with_uncertainty.csv", index=False)
    print(peak_df.to_string(index=False))

    langs_present = [l for l in C.LANG_ORDER if l in summ.lang.unique()]
    fig, axes = plt.subplots(1, len(langs_present), figsize=(4.2 * len(langs_present), 4.2), sharey=True)
    if len(langs_present) == 1:
        axes = [axes]
    colors = {"native": "tab:blue", "romanised": "tab:orange", "noncanonical": "tab:green"}
    for ax, lang in zip(axes, langs_present):
        sub = summ[summ.lang == lang]
        for cond in ["native", "romanised", "noncanonical"]:
            csub = sub[sub.condition == cond].sort_values("layer")
            ax.plot(csub.layer, csub.seed_mean_acc1, label=cond, color=colors[cond], marker="o", markersize=2)
            ax.fill_between(csub.layer, csub.seed_mean_acc1 - csub.seed_sd_acc1,
                            csub.seed_mean_acc1 + csub.seed_sd_acc1, color=colors[cond], alpha=0.2)
        bsub = df[(df.lang == lang) & (df.condition == "noncanonical") & (df.replicate_type == "bucket")]
        for alt_bucket, g in bsub.groupby("replicate_id"):
            g = g.sort_values("layer")
            ax.plot(g.layer, g.acc1, linestyle="--", linewidth=1, color="gray", alpha=0.7,
                   label=f"noncanonical @ {alt_bucket}")
        ax.set_title(lang.upper())
        ax.set_xlabel("layer")
    axes[0].set_ylabel("EN<->target retrieval acc@1 (seed mean +/- sd)")
    axes[-1].legend(loc="lower right", fontsize=6)
    fig.suptitle("layerwise probe (bootstrapped): cross-lingual retrieval acc@1 by layer, XLM-R-large, 3 seeds")
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / "layerwise_bootstrap.png", dpi=150)
    print(f"Figure written to {OUT_FIGURES / 'layerwise_bootstrap.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()

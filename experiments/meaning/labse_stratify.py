#!/usr/bin/env python3
"""Per-segment LaBSE retrieval and fragmentation change, for the tercile analysis.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import importlib.util
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

ORIG_REPO = Path.home() / "tokinv_work" / "repo" / "panel-pipeline"
NEW_REPO = Path.home() / "tokinv_work" / "repo" / "cluster_copy"
LABSE_PATH = Path.home() / "tokinv_work" / "staging" / "models" / "labse"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("constants_orig", ORIG_REPO / "src" / "constants.py")

N_PER_LANG = 200
SEED = C.SEED
BATCH_SIZE = 16


def words(text: str) -> list[str]:
    return unicodedata.normalize("NFC", text).split()


class LaBSEModel:
    def __init__(self, path: Path, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.encoder = AutoModel.from_pretrained(str(path)).to(device).eval()
        dense_sd = load_file(str(path / "2_Dense" / "model.safetensors"))
        self.dense = torch.nn.Linear(768, 768).to(device)
        self.dense.weight.data = dense_sd["linear.weight"].to(device)
        self.dense.bias.data = dense_sd["linear.bias"].to(device)
        self.device = device

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size=BATCH_SIZE) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                 max_length=256).to(self.device)
            hidden = self.encoder(**enc).last_hidden_state
            cls = hidden[:, 0, :]
            dense_out = torch.tanh(self.dense(cls))
            normed = torch.nn.functional.normalize(dense_out, p=2, dim=1)
            out.append(normed.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])


def retrieval_indicator(en: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    en_n = en / (np.linalg.norm(en, axis=1, keepdims=True) + 1e-9)
    tgt_n = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-9)
    sims = en_n @ tgt_n.T
    preds = sims.argmax(axis=1)
    return (preds == np.arange(len(preds))).astype(float)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid].copy()

    labse = LaBSEModel(LABSE_PATH, device)

    per_seg_rows = []
    for lang in C.LANG_ORDER:
        samples = {}
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond)]
            n = min(N_PER_LANG, len(sub))
            samp = sub.sample(n=n, random_state=SEED).reset_index(drop=True)
            samples[cond] = samp

        nat_ids = set(samples["native"]["segment_id"])
        rom_ids = set(samples["romanised"]["segment_id"])
        paired = nat_ids == rom_ids
        print(f"{lang}: native n={len(nat_ids)} romanised n={len(rom_ids)} paired={paired}")
        if not paired:
            print(f"  WARNING: {lang} not paired -- restrict to intersection")
            common = nat_ids & rom_ids
            for cond in C.CONDITIONS:
                samples[cond] = samples[cond][samples[cond].segment_id.isin(common)].reset_index(drop=True)

        embs = {}
        for cond in C.CONDITIONS:
            samp = samples[cond]
            en_emb = labse.embed(samp["source_en"].astype(str).tolist())
            tgt_emb = labse.embed(samp["target"].astype(str).tolist())
            correct = retrieval_indicator(en_emb, tgt_emb)
            embs[cond] = {"segment_id": samp["segment_id"].values, "correct": correct,
                         "mqm": samp["mqm_score"].values, "target": samp["target"].astype(str).tolist(),
                         "source_en": samp["source_en"].astype(str).tolist()}

        order = {sid: i for i, sid in enumerate(embs["romanised"]["segment_id"])}
        nat_ids_ordered = embs["native"]["segment_id"]
        idx_map = [order[sid] for sid in nat_ids_ordered]

        for i, sid in enumerate(nat_ids_ordered):
            j = idx_map[i]
            nat_tgt = embs["native"]["target"][i]
            rom_tgt = embs["romanised"]["target"][j]
            nat_src = embs["native"]["source_en"][i]
            n_tok_nat = labse.token_count(nat_tgt)
            n_tok_rom = labse.token_count(rom_tgt)
            n_wd_nat = len(words(nat_tgt))
            n_wd_rom = len(words(rom_tgt))
            n_tok_src = labse.token_count(nat_src)
            per_seg_rows.append({
                "lang": lang, "segment_id": sid,
                "native_correct": embs["native"]["correct"][i],
                "romanised_correct": embs["romanised"]["correct"][j],
                "mqm": embs["native"]["mqm"][i],
                "native_tp": n_tok_nat / n_tok_src if n_tok_src else np.nan,
                "romanised_tp": n_tok_rom / n_tok_src if n_tok_src else np.nan,
                "native_fertility": n_tok_nat / n_wd_nat if n_wd_nat else np.nan,
                "romanised_fertility": n_tok_rom / n_wd_rom if n_wd_rom else np.nan,
            })
        print(f"  {lang}: {len(nat_ids_ordered)} paired segments processed")

    df = pd.DataFrame(per_seg_rows)
    df["delta_tp"] = df["romanised_tp"] - df["native_tp"]
    df["delta_fertility"] = df["romanised_fertility"] - df["native_fertility"]
    out_path = NEW_REPO / "results" / "tables" / "labse_per_segment_fragmentation_retrieval.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWritten -> {out_path} ({len(df)} rows)")

    agg = df.groupby("lang").agg(
        native_acc1=("native_correct", "mean"),
        romanised_acc1=("romanised_correct", "mean"),
        n=("segment_id", "size"),
    )
    print("\nAggregate sanity check (compare to labse_bootstrap.csv):")
    print(agg)


if __name__ == "__main__":
    main()

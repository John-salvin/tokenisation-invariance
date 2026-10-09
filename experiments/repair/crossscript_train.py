"""Anchored adapter trained on Indic plus further scripts, holding one script out.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
sys.path.insert(0, str(ROB / "scripts"))
sys.path.insert(0, str(ROB / "scripts" / "lora"))
sys.path.insert(0, str(ROB / "src"))
import constants as C
import metrics_panel as MP
import lora_torch as L
import lora_lolo_pipeline as P
from anchored_train import train_anchored, SEED

NEW_LANGS = ["rus", "ara", "ces"]


def build_combined(held_out: str) -> pd.DataFrame:
    frames = []
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    e1 = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")
    indic_held = held_out if held_out in P.LANGS else "__none__"
    frames.append(P.build_lolo_training_examples(e1, seg, indic_held,
                                                 np.random.RandomState(SEED)))
    wmt = pd.read_parquet(C.DATA_INTERIM / "lora_train_wmt.parquet")
    wmt = wmt[wmt.lang != held_out]
    frames.append(wmt[["segment_id", "lang", "view", "src_ids", "mt_ids",
                       "ref_ids", "mqm_score"]])
    df = pd.concat(frames, ignore_index=True)

    df["mqm_score"] = df.groupby("lang")["mqm_score"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8))
    print("[labels] z-normalised within language; "
          f"global mean {df.mqm_score.mean():+.4f} sd {df.mqm_score.std():.4f}",
          flush=True)

    held_ids = set()
    if held_out in NEW_LANGS:
        allw = pd.read_parquet(C.DATA_INTERIM / "lora_train_wmt.parquet")
        held_ids = set(allw[allw.lang == held_out].segment_id)
    leaked = set(df.segment_id) & held_ids
    assert not leaked, f"LEAKAGE: {len(leaked)} held-out segment ids in training"
    print(f"[data] held_out={held_out!r}: {len(df)} rows, "
          f"{df.segment_id.nunique()} segments, langs {sorted(df.lang.unique())}",
          flush=True)
    print(df.groupby('lang').size().to_string(), flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out", required=True, choices=P.LANGS + NEW_LANGS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_sd", type=float, default=0.5)
    ap.add_argument("--lambda_anch", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--ckpt_every_steps", type=int, default=500)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    torch.manual_seed(SEED)
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = build_combined(a.held_out)

    model = MP.load_comet("comet22", device=device)
    replaced = L.inject_lora(model.encoder,
                             target_names=("query", "key", "value", "dense"),
                             r=a.r, alpha=a.alpha, path_filter=P.is_attention_linear)
    assert len(replaced) == 96, f"expected 96 attention linears, got {len(replaced)}"
    for n, p in model.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)
    model.to(device)

    train_anchored(model, train_df, device, out_dir, a.held_out,
                   a.epochs, a.batch_size, a.lr,
                   a.lambda_sd, a.lambda_anch, a.ckpt_every_steps)
    with open(out_dir / f"fold_{a.held_out}_DONE", "w") as f:
        json.dump({"held_out": a.held_out, "n_train_rows": len(train_df),
                   "langs": sorted(train_df.lang.unique())}, f)
    print("[done]", a.held_out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

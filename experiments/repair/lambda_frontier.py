"""Evaluate an adapter across the interpolation weight lambda.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
LIVE = SB_ROOT / "repo" / "panel-pipeline"
sys.path.insert(0, str(ROB / "scripts"))
sys.path.insert(0, str(ROB / "src"))
import constants as C
import noncanonical as NC
import tokenisers as T
import metrics_panel as MP
import lora_torch as L
from lora_lolo_pipeline import (MAIN_BUCKETS, TOKENISER, is_attention_linear,
                                score_eval_batch)

STAGE = Path(os.environ.get(
    "ADAPTER_ROOT", str(SB_ROOT / "staging" / "lora_lolo")))
CKPT_PAT = os.environ.get("ADAPTER_PATTERN", "checkpoints/lora_{held}_final.pt")
TAG = os.environ.get("OUT_TAG", "")
OUT = ROB / "results" / "tables"
LANGS = ["guj", "tam", "mal", "mar", "hin"]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
R = int(os.environ.get("LORA_R", "8"))
ALPHA = int(os.environ.get("LORA_ALPHA", str(2 * R)))
SEED = 20260828
PER_BUCKET = int(os.environ.get("PER_BUCKET", "250"))
BATCH = int(os.environ.get("BATCH", "48"))


def set_lambda(model, lam: float):
    n = 0
    for m in model.modules():
        if isinstance(m, L.LoRALinear):
            m.scaling = (ALPHA / R) * lam
            n += 1
    return n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg["lang"] = seg["lang"].astype(str); seg["condition"] = seg["condition"].astype(str)
    e1 = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")

    model = MP.load_comet("comet22", device=device)
    replaced = L.inject_lora(model.encoder, target_names=("query", "key", "value", "dense"),
                             r=R, alpha=ALPHA, path_filter=is_attention_linear)
    assert len(replaced) == 96, f"expected 96 attention linears, got {len(replaced)}"
    model.to(device).eval()
    print(f"[model] LoRA injected into {len(replaced)} layers", flush=True)

    rows = []
    for held in LANGS:
        ckpt = STAGE / held / CKPT_PAT.format(held=held)
        if not ckpt.exists():
            print(f"[skip] no adapter for {held}")
            continue
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        for k, v in state["adapter"].items():
            if k.endswith("lora_A"):
                saved_r = v.shape[0]
                if saved_r != R:
                    raise SystemExit(
                        f"rank mismatch: {ckpt} was trained at r={saved_r}, "
                        f"evaluator injected r={R}. Set LORA_R={saved_r}.")
                break
        L.load_lora_state_dict(model, state["adapter"], device=device)
        model.to(device).eval()
        print(f"\n== fold held_out={held}: loaded {ckpt.name} "
              f"(step={state.get('step')})", flush=True)

        nat = seg[(seg.lang == held) & (seg.condition == "native") & seg.mqm_valid]
        nat_recs = [{"src_ids": T.tokenise(str(r.source_en), TOKENISER, add_special_tokens=False),
                     "mt_ids": T.tokenise(NC.normalize_text_for_reseg(str(r.target)), TOKENISER,
                                          add_special_tokens=False),
                     "ref_ids": T.tokenise(str(r.reference), TOKENISER, add_special_tokens=False)}
                    for _, r in nat.iterrows()]
        nat_mqm = nat.mqm_score.to_numpy()
        base_rho_stored = sps.spearmanr(nat.comet_22, nat.mqm_score)[0]

        held_fwd = e1[(e1.direction == "forward") & (e1.lang == held)
                      & (e1.bucket.isin(MAIN_BUCKETS)) & e1.mt_ids.notna()]
        held_fwd = held_fwd.groupby("bucket", observed=True, group_keys=False).apply(
            lambda g: g.sample(n=min(PER_BUCKET, len(g)), random_state=SEED))
        segi = seg[seg.condition == "native"].set_index("segment_id")
        reseg_recs, reseg_meta = [], []
        for _, r in held_fwd.iterrows():
            s = segi.loc[r.segment_id]
            reseg_recs.append({"src_ids": T.tokenise(str(s.source_en), TOKENISER, add_special_tokens=False),
                            "mt_ids": list(r.mt_ids),
                            "ref_ids": T.tokenise(str(s.reference), TOKENISER, add_special_tokens=False)})
            reseg_meta.append({"bucket": r.bucket, "mqm_score": s.mqm_score})
        reseg_meta = pd.DataFrame(reseg_meta)
        print(f"   eval: {len(nat_recs)} canonical, {len(reseg_recs)} re-segmentation samples", flush=True)

        for lam in LAMBDAS:
            set_lambda(model, lam)
            with torch.no_grad():
                nat_s = score_eval_batch(model, nat_recs, device, batch_size=BATCH)
                reseg_s = score_eval_batch(model, reseg_recs, device, batch_size=BATCH)
            rho_canon = sps.spearmanr(nat_s, nat_mqm)[0]
            m = reseg_meta.copy(); m["score"] = reseg_s
            per_bucket = {}
            for b in MAIN_BUCKETS:
                sb = m[m.bucket == b]
                per_bucket[b] = (sps.spearmanr(sb.score, sb.mqm_score)[0]
                                 if len(sb) > 2 and sb.score.nunique() > 1 else float("nan"))
            rows.append({"held_out": held, "lambda": lam, "rho_canonical": rho_canon,
                         "rho_most_fragmented": per_bucket[MAIN_BUCKETS[-1]],
                         "rho_mildest": per_bucket[MAIN_BUCKETS[0]],
                         "base_rho_canonical_stored": base_rho_stored,
                         "n_canonical": len(nat_recs), "n_e1": len(reseg_recs),
                         **{f"rho_{b}": v for b, v in per_bucket.items()}})
            print(f"   lambda={lam:.2f}  canonical={rho_canon:.4f}  "
                  f"most_frag={per_bucket[MAIN_BUCKETS[-1]]:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"lora_anchored_frontier{TAG}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"lora_anchored_frontier{TAG}.csv", index=False)

    an = []
    for held, g in df.groupby("held_out"):
        g = g.set_index("lambda")
        c0, f0 = g.loc[0.0, "rho_canonical"], g.loc[0.0, "rho_most_fragmented"]
        c1, f1 = g.loc[1.0, "rho_canonical"], g.loc[1.0, "rho_most_fragmented"]
        full_gain = f1 - f0
        for lam in LAMBDAS:
            gain = g.loc[lam, "rho_most_fragmented"] - f0
            closs = c0 - g.loc[lam, "rho_canonical"]
            an.append({"held_out": held, "lambda": lam,
                       "frag_gain": gain, "frag_gain_retained_pct":
                           100 * gain / full_gain if full_gain else float("nan"),
                       "canonical_loss": closs,
                       "pareto_ok": bool(full_gain and gain / full_gain >= 0.70 and closs <= 0.02)})
    adf = pd.DataFrame(an)
    adf.to_csv(OUT / f"lora_lambda_frontier_analysis{TAG}.csv", index=False)
    print("\n== frontier analysis ==")
    print(adf.round(4).to_string(index=False))
    print("\n== lambda=0 sanity (must match stored base COMET-22 Spearman) ==")
    z = df[df["lambda"] == 0.0][["held_out", "rho_canonical", "base_rho_canonical_stored"]]
    z["abs_diff"] = (z.rho_canonical - z.base_rho_canonical_stored).abs()
    print(z.round(5).to_string(index=False))


if __name__ == "__main__":
    main()

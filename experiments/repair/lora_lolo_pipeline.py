#!/usr/bin/env python3
"""Leave-one-language-out training and evaluation of the plain multi-tokenisation adapter (A1).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import metrics_panel as MP
import noncanonical as NC
import tokenisers as T
import lora_torch as L

LANGS = ["guj", "tam", "mal", "mar", "hin"]
MAIN_BUCKETS = ["[1.00,1.25)", "[1.25,1.50)", "[1.50,1.75)", "[1.75,2.00)",
                "[2.00,2.25)", "[2.25,2.50)", "[2.50,2.75)", "[2.75,3.00)"]
TOKENISER = "xlmr"
SEED = 20260828


def is_attention_linear(full_name: str) -> bool:
    return "attention" in full_name


def build_lolo_training_examples(reseg_samples: pd.DataFrame, seg: pd.DataFrame,
                                  held_out: str, rng: np.random.RandomState) -> pd.DataFrame:
    train_langs = [l for l in LANGS if l != held_out]
    seg_nat = seg[(seg.condition == "native") & (seg.lang.isin(train_langs)) & seg.mqm_valid].copy()

    fwd = reseg_samples[(reseg_samples.direction == "forward") & (reseg_samples.lang.isin(train_langs))]
    held_seg_ids = set(seg[seg.lang == held_out].segment_id)

    rows = []
    for seg_id, grp in fwd.groupby("segment_id", observed=True):
        srow = seg_nat[seg_nat.segment_id == seg_id]
        if srow.empty:
            continue
        srow = srow.iloc[0]
        lang = srow.lang
        mqm = srow.mqm_score
        src_ids = T.tokenise(str(srow.source_en), TOKENISER, add_special_tokens=False)
        ref_ids = T.tokenise(str(srow.reference), TOKENISER, add_special_tokens=False)

        canon_text = NC.normalize_text_for_reseg(str(srow.target))
        canon_ids = T.tokenise(canon_text, TOKENISER, add_special_tokens=False)
        rows.append({"segment_id": seg_id, "lang": lang, "view": "canonical",
                     "src_ids": src_ids, "mt_ids": canon_ids, "ref_ids": ref_ids, "mqm_score": mqm})

        char_row = grp[grp.bucket == NC.CHAR_LEVEL_BUCKET]
        char_row = char_row[char_row.mt_ids.notna()]
        if len(char_row):
            rows.append({"segment_id": seg_id, "lang": lang, "view": "char_level",
                         "src_ids": src_ids, "mt_ids": list(char_row.iloc[0].mt_ids),
                         "ref_ids": ref_ids, "mqm_score": mqm})

        for b in MAIN_BUCKETS:
            br = grp[(grp.bucket == b) & (grp.seed == 0)]
            br = br[br.mt_ids.notna()]
            if len(br):
                rows.append({"segment_id": seg_id, "lang": lang, "view": f"noncanon_{b}",
                             "src_ids": src_ids, "mt_ids": list(br.iloc[0].mt_ids),
                             "ref_ids": ref_ids, "mqm_score": mqm})

    train_df = pd.DataFrame(rows)

    leaked = set(train_df.segment_id) & held_seg_ids
    print(f"[leakage check] held-out lang = {held_out!r}, held-out segment_ids = {len(held_seg_ids)}, "
          f"training rows = {len(train_df)}, training unique segment_ids = {train_df.segment_id.nunique()}")
    print(f"[leakage check] intersection(training segment_ids, held-out segment_ids) = {len(leaked)}")
    assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} held-out segment_ids found in training set: {sorted(leaked)[:5]}..."
    assert train_df.lang.isin(train_langs).all(), "non-training-language row found in training set"
    assert not (train_df.lang == held_out).any(), "held-out-language row found in training set"
    print(f"[leakage check] PASS -- zero leakage confirmed for fold held_out={held_out!r}")
    return train_df


def make_batches(df: pd.DataFrame, batch_size: int, rng: np.random.RandomState):
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield df.loc[idx[start:start + batch_size]]


def to_side_tensors(model, ids_list: list[list[int]], device):
    tok = model.encoder.tokenizer
    bos, eos = tok.bos_token_id, tok.eos_token_id
    seqs = [[bos] + list(ids) + [eos] for ids in ids_list]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 1
    input_ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    return input_ids.to(device), attn.to(device)


def score_batch_trainable(model, batch: pd.DataFrame, device) -> torch.Tensor:
    src_ii, src_am = to_side_tensors(model, batch.src_ids.tolist(), device)
    mt_ii, mt_am = to_side_tensors(model, batch.mt_ids.tolist(), device)
    ref_ii, ref_am = to_side_tensors(model, batch.ref_ids.tolist(), device)
    out = model.forward(src_ii, src_am, mt_ii, mt_am, ref_ii, ref_am)
    return out.score


def train_fold(model, train_df: pd.DataFrame, device, out_dir: Path, held_out: str,
               epochs: int, batch_size: int, lr: float, ckpt_every_steps: int) -> list[dict]:
    target = torch.tensor(train_df.mqm_score.to_numpy(dtype=np.float32))
    mu, sigma = target.mean().item(), target.std().item()
    print(f"[train] target mqm_score: mean={mu:.3f} std={sigma:.3f} (z-normalised for the regression loss; "
          f"eval uses Spearman, which is scale-invariant, so this normalisation choice doesn't affect eval)")

    trainable = L.trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable (LoRA-only) params: {n_trainable:,} / {n_total:,} total "
          f"({100 * n_trainable / n_total:.3f}%)")
    opt = torch.optim.AdamW(trainable, lr=lr)

    rng = np.random.RandomState(SEED)
    loss_log = []
    step = 0
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(epochs):
        for batch in make_batches(train_df, batch_size, rng):
            opt.zero_grad()
            scores = score_batch_trainable(model, batch, device)
            y = (torch.tensor(batch.mqm_score.to_numpy(dtype=np.float32), device=device) - mu) / (sigma + 1e-8)
            loss = torch.nn.functional.mse_loss(scores, y)
            loss.backward()
            opt.step()
            step += 1
            loss_log.append({"fold_held_out": held_out, "epoch": epoch, "step": step,
                              "loss": loss.item(), "elapsed_s": time.time() - t0})
            if step % 20 == 0:
                print(f"[train] fold={held_out} epoch={epoch} step={step} loss={loss.item():.4f}")
            if step % ckpt_every_steps == 0:
                ckpt_path = ckpt_dir / f"lora_{held_out}_step{step}.pt"
                torch.save({"adapter": L.lora_state_dict(model), "mu": mu, "sigma": sigma,
                            "step": step, "epoch": epoch}, ckpt_path)
                print(f"[checkpoint] saved {ckpt_path}")
    final_path = ckpt_dir / f"lora_{held_out}_final.pt"
    torch.save({"adapter": L.lora_state_dict(model), "mu": mu, "sigma": sigma,
                "step": step, "epoch": epochs - 1}, final_path)
    print(f"[checkpoint] final saved {final_path}")
    pd.DataFrame(loss_log).to_csv(out_dir / f"loss_curve_{held_out}.csv", index=False)
    return loss_log


@torch.no_grad()
def score_eval_batch(model, ids_batches, device, batch_size=32) -> list[float]:
    df = pd.DataFrame(ids_batches)
    out_scores = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        scores = score_batch_trainable(model, batch, device)
        out_scores.extend(scores.detach().cpu().tolist())
    return out_scores


def evaluate_fold(model, held_out: str, seg: pd.DataFrame, reseg_samples: pd.DataFrame,
                   reseg_scores_baseline: pd.DataFrame, out_dir: Path, device, max_eval_rows=None) -> dict:
    model.eval()
    results = {}

    held_fwd = reseg_samples[(reseg_samples.direction == "forward") & (reseg_samples.lang == held_out)
                           & (reseg_samples.bucket.isin(MAIN_BUCKETS)) & reseg_samples.mt_ids.notna()]
    if max_eval_rows is not None:
        held_fwd = held_fwd.groupby("bucket", observed=True, group_keys=False).apply(
            lambda g: g.sample(n=min(max(2, max_eval_rows // len(MAIN_BUCKETS)), len(g)), random_state=SEED))
    seg_by_id = seg.set_index(["segment_id", "condition"])
    rows = []
    for _, r in held_fwd.iterrows():
        srow = seg_by_id.loc[(r.segment_id, "native")]
        src_ids = T.tokenise(str(srow.source_en), TOKENISER, add_special_tokens=False)
        ref_ids = T.tokenise(str(srow.reference), TOKENISER, add_special_tokens=False)
        rows.append({"segment_id": r.segment_id, "bucket": r.bucket, "seed": r.seed,
                     "src_ids": src_ids, "mt_ids": list(r.mt_ids), "ref_ids": ref_ids,
                     "mqm_score": srow.mqm_score})
    eval_df = pd.DataFrame(rows)
    lora_scores = score_eval_batch(model, eval_df.to_dict("records"), device)
    eval_df["lora_score"] = lora_scores

    dose_rows = []
    for b in MAIN_BUCKETS:
        s = eval_df[eval_df.bucket == b]
        base = reseg_scores_baseline[(reseg_scores_baseline.lang == held_out) & (reseg_scores_baseline.direction == "forward")
                                   & (reseg_scores_baseline.bucket == b)]
        rho_lora = sps.spearmanr(s.lora_score, s.mqm_score)[0] if s.lora_score.nunique() > 1 else float("nan")
        rho_base = sps.spearmanr(base.comet22, base.mqm_score)[0] if base.comet22.nunique() > 1 else float("nan")
        dose_rows.append({"held_out_lang": held_out, "bucket": b, "n_lora": len(s), "n_baseline": len(base),
                           "rho_baseline_comet22": rho_base, "rho_lora": rho_lora,
                           "delta_rho": rho_lora - rho_base if pd.notna(rho_lora) and pd.notna(rho_base) else float("nan")})
    dose_df = pd.DataFrame(dose_rows)
    results["dose_response"] = dose_df
    print(f"\n[eval] fold={held_out} re-segmentation dose-response (baseline COMET-22 vs LoRA), per bucket:")
    print(dose_df.to_string(index=False))

    nr_rows = []
    for cond in ["native", "romanised"]:
        s = seg[(seg.lang == held_out) & (seg.condition == cond) & seg.mqm_valid].copy()
        if max_eval_rows is not None:
            s = s.sample(n=min(max_eval_rows, len(s)), random_state=SEED)
        recs = []
        for _, r in s.iterrows():
            recs.append({"src_ids": T.tokenise(str(r.source_en), TOKENISER, add_special_tokens=False),
                         "mt_ids": T.tokenise(NC.normalize_text_for_reseg(str(r.target)) if cond == "native" else str(r.target),
                                              TOKENISER, add_special_tokens=False),
                         "ref_ids": T.tokenise(str(r.reference), TOKENISER, add_special_tokens=False),
                         "mqm_score": r.mqm_score})
        lora_s = score_eval_batch(model, recs, device)
        rho_lora = sps.spearmanr(lora_s, s.mqm_score)[0]
        rho_base = sps.spearmanr(s.comet_22, s.mqm_score)[0]
        nr_rows.append({"held_out_lang": held_out, "condition": cond, "n": len(s),
                         "rho_baseline_comet22": rho_base, "rho_lora": rho_lora,
                         "delta_rho": rho_lora - rho_base})
    nr_df = pd.DataFrame(nr_rows)
    results["native_romanised"] = nr_df
    print(f"\n[eval] fold={held_out} native/romanised comparison:")
    print(nr_df.to_string(index=False))

    canon_row = nr_df[nr_df.condition == "native"].iloc[0]
    canon_rows = [{"held_out_lang": held_out, "n": int(canon_row.n),
                   "rho_baseline_comet22_canonical": canon_row.rho_baseline_comet22,
                   "rho_lora_canonical": canon_row.rho_lora,
                   "delta_rho": canon_row.rho_lora - canon_row.rho_baseline_comet22,
                   "regressed": bool(canon_row.rho_lora < canon_row.rho_baseline_comet22)}]
    canon_df = pd.DataFrame(canon_rows)
    results["canonical_regression"] = canon_df
    print(f"\n[eval] fold={held_out} canonical-accuracy regression check:")
    print(canon_df.to_string(index=False))

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out", required=True, choices=LANGS)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_every_steps", type=int, default=200)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_train_rows", type=int, default=None, help="smoke-test cap, not for real runs")
    ap.add_argument("--max_eval_rows", type=int, default=None, help="smoke-test cap, not for real runs")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} fold_held_out={args.held_out}")

    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    reseg_samples = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")
    reseg_scores_baseline = pd.read_csv(ROOT / "results" / "tables" / "reseg_scores.csv")

    rng = np.random.RandomState(SEED)
    train_df = build_lolo_training_examples(reseg_samples, seg, args.held_out, rng)
    print(f"[data] fold={args.held_out}: {len(train_df)} training rows, "
          f"{train_df.segment_id.nunique()} unique segments, "
          f"languages={sorted(train_df.lang.unique())}, views/segment~{len(train_df)/max(1,train_df.segment_id.nunique()):.1f}")
    if args.max_train_rows is not None:
        train_df = train_df.sample(n=min(args.max_train_rows, len(train_df)), random_state=SEED).reset_index(drop=True)
        print(f"[SMOKE TEST] capped training rows to {len(train_df)}")
    train_df.to_parquet(out_dir / f"lolo_train_{args.held_out}.parquet", index=False)

    model = MP.load_comet("comet22", device=device)
    replaced = L.inject_lora(model.encoder, target_names=("query", "key", "value", "dense"),
                              r=args.r, alpha=args.alpha, path_filter=is_attention_linear)
    print(f"[model] LoRA injected into {len(replaced)} attention linear layers "
          f"(expected 4/layer x 24 layers = 96): {replaced[:4]} ... {replaced[-4:]}")
    assert len(replaced) == 96, f"expected 96 attention linear layers, got {len(replaced)} -- inspect path_filter"

    for name, p in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            p.requires_grad_(False)
    model.to(device)

    loss_log = train_fold(model, train_df, device, out_dir, args.held_out,
                           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                           ckpt_every_steps=args.ckpt_every_steps)

    eval_results = evaluate_fold(model, args.held_out, seg, reseg_samples, reseg_scores_baseline, out_dir, device,
                                  max_eval_rows=args.max_eval_rows)
    eval_results["dose_response"].to_csv(out_dir / f"dose_response_{args.held_out}.csv", index=False)
    eval_results["native_romanised"].to_csv(out_dir / f"native_romanised_{args.held_out}.csv", index=False)
    eval_results["canonical_regression"].to_csv(out_dir / f"canonical_regression_{args.held_out}.csv", index=False)

    with open(out_dir / f"fold_{args.held_out}_DONE", "w") as f:
        json.dump({"held_out": args.held_out, "n_train": len(train_df), "n_steps": len(loss_log)}, f)
    print(f"\n[done] fold={args.held_out} complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the anchored multi-tokenisation adapter (A2).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import argparse, json, sys, time
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
sys.path.insert(0, str(ROB / "scripts"))
sys.path.insert(0, str(ROB / "src"))
import constants as C
import metrics_panel as MP
import lora_torch as L
import lora_lolo_pipeline as P

SEED = P.SEED


def set_lambda(model, lam: float):
    for m in model.modules():
        if isinstance(m, L.LoRALinear):
            if not hasattr(m, "_base_scaling"):
                m._base_scaling = m.scaling
            m.scaling = m._base_scaling * lam


@torch.no_grad()
def teacher_canonical_scores(model, canon_rows: pd.DataFrame, device, batch_size=32):
    set_lambda(model, 0.0)
    out = []
    for s in range(0, len(canon_rows), batch_size):
        b = canon_rows.iloc[s:s + batch_size]
        out.extend(P.score_batch_trainable(model, b, device).detach().cpu().tolist())
    set_lambda(model, 1.0)
    return np.asarray(out, dtype=np.float32)


def train_anchored(model, train_df, device, out_dir: Path, held_out: str,
                   epochs: int, batch_size: int, lr: float,
                   lambda_sd: float, lambda_anch: float, ckpt_every_steps: int,
                   max_seconds: float | None = None):
    target = torch.tensor(train_df.mqm_score.to_numpy(dtype=np.float32))
    mu, sigma = target.mean().item(), target.std().item()

    canon = train_df[train_df.view == "canonical"].drop_duplicates("segment_id").reset_index(drop=True)
    canon_by_seg = canon.set_index("segment_id")
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    resume_path = ck / f"ckpt_resume_{held_out}.pt"

    trainable = L.trainable_parameters(model)
    print(f"[train] trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = np.random.RandomState(SEED)

    start_epoch, skip_batches, step, log = 0, 0, 0, []
    state = None
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
    if state is not None:
        L.load_lora_state_dict(model, state["adapter"], device=device)
        opt.load_state_dict(state["optimizer"])
        rng.set_state(state["np_rng_state_epoch_start"])
        torch.set_rng_state(state["torch_rng_state"].cpu()
                            if hasattr(state["torch_rng_state"], "cpu")
                            else state["torch_rng_state"])
        teacher = state["teacher"]
        start_epoch = state["epoch"]
        skip_batches = state["batches_done_in_epoch"]
        step = state["step"]
        log = state["log"]
        mu, sigma = state["mu"], state["sigma"]
        print(f"[resume] {resume_path.name}: epoch {start_epoch}, step {step}, "
              f"skipping {skip_batches} batches of this epoch", flush=True)
    else:
        print(f"[teacher] scoring {len(canon)} canonical views with the frozen base",
              flush=True)
        t_scores = teacher_canonical_scores(model, canon, device)
        teacher = dict(zip(canon.segment_id, t_scores))
        print(f"[teacher] done: mean={t_scores.mean():.4f} std={t_scores.std():.4f}",
              flush=True)

    t0 = time.time()
    elapsed_before = log[-1]["elapsed_s"] if log else 0.0

    def save_resume(epoch: int, batches_done: int) -> None:
        tmp = resume_path.with_suffix(".pt.tmp")
        torch.save({"adapter": L.lora_state_dict(model),
                    "optimizer": opt.state_dict(),
                    "np_rng_state_epoch_start": rng_state_epoch_start,
                    "torch_rng_state": torch.get_rng_state(),
                    "teacher": teacher, "mu": mu, "sigma": sigma,
                    "epoch": epoch, "batches_done_in_epoch": batches_done,
                    "step": step, "log": log}, tmp)
        tmp.replace(resume_path)

    stopped_early = False
    for epoch in range(start_epoch, epochs):
        rng_state_epoch_start = rng.get_state()
        batches_done = 0
        for batch in P.make_batches(train_df, batch_size, rng):
            if epoch == start_epoch and batches_done < skip_batches:
                batches_done += 1
                continue
            opt.zero_grad()
            scores = P.score_batch_trainable(model, batch, device)
            y = (torch.tensor(batch.mqm_score.to_numpy(dtype=np.float32), device=device) - mu) / (sigma + 1e-8)
            l_reg = torch.nn.functional.mse_loss(scores, y)

            tvec = torch.tensor([teacher[s] for s in batch.segment_id],
                                dtype=torch.float32, device=device)
            frag = torch.tensor((batch.view != "canonical").to_numpy(), device=device)
            if frag.any():
                l_sd = torch.nn.functional.mse_loss(scores[frag], tvec[frag])
            else:
                l_sd = torch.zeros((), device=device)

            segs = list(dict.fromkeys(batch.segment_id.tolist()))
            cb = canon_by_seg.loc[segs].reset_index()
            s_canon = P.score_batch_trainable(model, cb, device)
            t_canon = torch.tensor([teacher[s] for s in segs],
                                   dtype=torch.float32, device=device)
            l_anch = torch.nn.functional.mse_loss(s_canon, t_canon)

            loss = l_reg + lambda_sd * l_sd + lambda_anch * l_anch
            loss.backward()
            opt.step()
            step += 1
            batches_done += 1
            log.append({"fold_held_out": held_out, "epoch": epoch, "step": step,
                        "loss": loss.item(), "l_reg": l_reg.item(),
                        "l_sd": float(l_sd.item()), "l_anch": l_anch.item(),
                        "elapsed_s": elapsed_before + time.time() - t0})
            if step % 25 == 0:
                print(f"[train] {held_out} e{epoch} s{step} loss={loss.item():.4f} "
                      f"(reg={l_reg.item():.4f} sd={float(l_sd.item()):.4f} "
                      f"anch={l_anch.item():.4f})", flush=True)
            if ckpt_every_steps and step % ckpt_every_steps == 0:
                save_resume(epoch, batches_done)
                print(f"[ckpt] step {step} -> {resume_path.name}", flush=True)
            if max_seconds is not None and (time.time() - t0) > max_seconds:
                save_resume(epoch, batches_done)
                print(f"[budget] soft limit {max_seconds:.0f}s reached at step {step}; "
                      f"checkpoint written, exiting for the next job in the chain",
                      flush=True)
                stopped_early = True
                break
        if stopped_early:
            break

    if stopped_early:
        pd.DataFrame(log).to_csv(out_dir / f"loss_curve_anchored_{held_out}.csv",
                                 index=False)
        return log, False

    torch.save({"adapter": L.lora_state_dict(model), "mu": mu, "sigma": sigma,
                "step": step, "lambda_sd": lambda_sd, "lambda_anch": lambda_anch},
               ck / f"lora_anchored_{held_out}_final.pt")
    pd.DataFrame(log).to_csv(out_dir / f"loss_curve_anchored_{held_out}.csv", index=False)
    if resume_path.exists():
        resume_path.unlink()
    return log, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_out", required=True, choices=P.LANGS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_sd", type=float, default=0.5)
    ap.add_argument("--lambda_anch", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--ckpt_every_steps", type=int, default=500)
    ap.add_argument("--max_seconds", type=float, default=None,
                    help="soft wall budget. On expiry write the resume checkpoint "
                         "and exit 0, so a chained job can continue the fold.")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    e1 = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")
    train_df = P.build_lolo_training_examples(e1, seg, args.held_out,
                                              np.random.RandomState(SEED))
    print(f"[data] {len(train_df)} rows, {train_df.segment_id.nunique()} segments", flush=True)

    model = MP.load_comet("comet22", device=device)
    replaced = L.inject_lora(model.encoder, target_names=("query", "key", "value", "dense"),
                             r=args.r, alpha=args.alpha, path_filter=P.is_attention_linear)
    assert len(replaced) == 96, f"expected 96, got {len(replaced)}"
    for n, p in model.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)
    model.to(device)

    done_marker = out_dir / f"fold_{args.held_out}_DONE"
    if done_marker.exists():
        print(f"[skip] {done_marker.name} already present; nothing to do")
        return

    _log, finished = train_anchored(
        model, train_df, device, out_dir, args.held_out,
        args.epochs, args.batch_size, args.lr,
        args.lambda_sd, args.lambda_anch, args.ckpt_every_steps,
        max_seconds=args.max_seconds)
    if not finished:
        print("[incomplete]", args.held_out, "- resume checkpoint written")
        return
    with open(done_marker, "w") as f:
        json.dump({"held_out": args.held_out, "lambda_sd": args.lambda_sd,
                   "lambda_anch": args.lambda_anch}, f)
    print("[done]", args.held_out)


if __name__ == "__main__":
    main()

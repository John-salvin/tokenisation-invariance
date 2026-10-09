"""Inner-fold adapters for the nested selection of lambda.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats as sps

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB_ROOT / "repo" / "cluster_copy"
sys.path.insert(0, str(ROB / "scripts"))
sys.path.insert(0, str(ROB / "src"))
import constants as C
import noncanonical as NC
import tokenisers as T
import metrics_panel as MP
import lora_torch as L
import lora_lolo_pipeline as P
from lora_lolo_pipeline import MAIN_BUCKETS, TOKENISER, is_attention_linear, score_eval_batch

LANGS = P.LANGS
SEED = P.SEED
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
R, ALPHA = 8, 16
PER_BUCKET = int(os.environ.get("PER_BUCKET", "1500"))
BATCH = int(os.environ.get("BATCH", "48"))
OUT = ROB / "results" / "tables" / "nested"


def build_training_examples(e1, seg, train_langs, excluded, rng):
    seg_nat = seg[(seg.condition == "native") & (seg.lang.isin(train_langs)) & seg.mqm_valid].copy()
    fwd = e1[(e1.direction == "forward") & (e1.lang.isin(train_langs))]

    rows = []
    for seg_id, grp in fwd.groupby("segment_id", observed=True):
        srow = seg_nat[seg_nat.segment_id == seg_id]
        if srow.empty:
            continue
        srow = srow.iloc[0]
        lang, mqm = srow.lang, srow.mqm_score
        src_ids = T.tokenise(str(srow.source_en), TOKENISER, add_special_tokens=False)
        ref_ids = T.tokenise(str(srow.reference), TOKENISER, add_special_tokens=False)

        canon_ids = T.tokenise(NC.normalize_text_for_reseg(str(srow.target)), TOKENISER,
                               add_special_tokens=False)
        rows.append({"segment_id": seg_id, "lang": lang, "view": "canonical",
                     "src_ids": src_ids, "mt_ids": canon_ids, "ref_ids": ref_ids,
                     "mqm_score": mqm})

        ch = grp[(grp.bucket == NC.CHAR_LEVEL_BUCKET) & grp.mt_ids.notna()]
        if len(ch):
            rows.append({"segment_id": seg_id, "lang": lang, "view": "char_level",
                         "src_ids": src_ids, "mt_ids": list(ch.iloc[0].mt_ids),
                         "ref_ids": ref_ids, "mqm_score": mqm})

        for b in MAIN_BUCKETS:
            br = grp[(grp.bucket == b) & (grp.seed == 0) & grp.mt_ids.notna()]
            if len(br):
                rows.append({"segment_id": seg_id, "lang": lang, "view": f"noncanon_{b}",
                             "src_ids": src_ids, "mt_ids": list(br.iloc[0].mt_ids),
                             "ref_ids": ref_ids, "mqm_score": mqm})

    df = pd.DataFrame(rows)

    for ex in excluded:
        ex_ids = set(seg[seg.lang == ex].segment_id)
        leaked = set(df.segment_id) & ex_ids
        print(f"[leakage] excluded={ex!r}: {len(ex_ids)} segment_ids, "
              f"intersection with training = {len(leaked)}", flush=True)
        assert not leaked, f"LEAKAGE of {ex}: {sorted(leaked)[:5]}"
        assert not (df.lang == ex).any(), f"{ex} rows present in training set"
    assert df.lang.isin(train_langs).all()
    print(f"[leakage] PASS -- train_langs={train_langs}, excluded={excluded}, "
          f"{len(df)} rows, {df.segment_id.nunique()} segments", flush=True)
    return df


def set_lambda(model, lam: float):
    for m in model.modules():
        if isinstance(m, L.LoRALinear):
            m.scaling = (ALPHA / R) * lam


@torch.no_grad()
def teacher_canonical_scores(model, canon_rows, device, batch_size=32):
    set_lambda(model, 0.0)
    out = []
    for s in range(0, len(canon_rows), batch_size):
        out.extend(P.score_batch_trainable(model, canon_rows.iloc[s:s + batch_size],
                                           device).detach().cpu().tolist())
    set_lambda(model, 1.0)
    return np.asarray(out, dtype=np.float32)


def train_anchored(model, train_df, device, tag, epochs, batch_size, lr,
                   lambda_sd, lambda_anch):
    target = torch.tensor(train_df.mqm_score.to_numpy(dtype=np.float32))
    mu, sigma = target.mean().item(), target.std().item()

    canon = train_df[train_df.view == "canonical"].drop_duplicates("segment_id").reset_index(drop=True)
    t_scores = teacher_canonical_scores(model, canon, device)
    teacher = dict(zip(canon.segment_id, t_scores))
    canon_by_seg = canon.set_index("segment_id")
    print(f"[teacher] {len(canon)} canonical views, mean={t_scores.mean():.4f}", flush=True)

    trainable = L.trainable_parameters(model)
    opt = torch.optim.AdamW(trainable, lr=lr)
    rng = np.random.RandomState(SEED)
    log, step, t0 = [], 0, time.time()

    for epoch in range(epochs):
        for batch in P.make_batches(train_df, batch_size, rng):
            opt.zero_grad()
            scores = P.score_batch_trainable(model, batch, device)
            y = (torch.tensor(batch.mqm_score.to_numpy(dtype=np.float32), device=device) - mu) / (sigma + 1e-8)
            l_reg = torch.nn.functional.mse_loss(scores, y)

            tvec = torch.tensor([teacher[s] for s in batch.segment_id],
                                dtype=torch.float32, device=device)
            frag = torch.tensor((batch.view != "canonical").to_numpy(), device=device)
            l_sd = (torch.nn.functional.mse_loss(scores[frag], tvec[frag])
                    if frag.any() else torch.zeros((), device=device))

            segs = list(dict.fromkeys(batch.segment_id.tolist()))
            cb = canon_by_seg.loc[segs].reset_index()
            s_canon = P.score_batch_trainable(model, cb, device)
            t_canon = torch.tensor([teacher[s] for s in segs], dtype=torch.float32, device=device)
            l_anch = torch.nn.functional.mse_loss(s_canon, t_canon)

            loss = l_reg + lambda_sd * l_sd + lambda_anch * l_anch
            loss.backward()
            opt.step()
            step += 1
            log.append({"tag": tag, "epoch": epoch, "step": step, "loss": loss.item(),
                        "l_reg": l_reg.item(), "l_sd": float(l_sd.item()),
                        "l_anch": l_anch.item(), "elapsed_s": time.time() - t0})
            if step % 50 == 0:
                print(f"[train] {tag} e{epoch} s{step} loss={loss.item():.4f} "
                      f"(reg={l_reg.item():.4f} sd={float(l_sd.item()):.4f} "
                      f"anch={l_anch.item():.4f})", flush=True)
    return log, mu, sigma, step


def evaluate_on(model, val_lang, seg, e1, device):
    nat = seg[(seg.lang == val_lang) & (seg.condition == "native") & seg.mqm_valid]
    nat_recs = [{"src_ids": T.tokenise(str(r.source_en), TOKENISER, add_special_tokens=False),
                 "mt_ids": T.tokenise(NC.normalize_text_for_reseg(str(r.target)), TOKENISER,
                                      add_special_tokens=False),
                 "ref_ids": T.tokenise(str(r.reference), TOKENISER, add_special_tokens=False)}
                for _, r in nat.iterrows()]
    nat_mqm = nat.mqm_score.to_numpy()
    base_rho_stored = sps.spearmanr(nat.comet_22, nat.mqm_score)[0]

    fwd = e1[(e1.direction == "forward") & (e1.lang == val_lang)
             & (e1.bucket.isin(MAIN_BUCKETS)) & e1.mt_ids.notna()]
    fwd = fwd.groupby("bucket", observed=True, group_keys=False).apply(
        lambda g: g.sample(n=min(PER_BUCKET, len(g)), random_state=SEED))
    segi = seg[seg.condition == "native"].set_index("segment_id")
    recs, meta = [], []
    for _, r in fwd.iterrows():
        s = segi.loc[r.segment_id]
        recs.append({"src_ids": T.tokenise(str(s.source_en), TOKENISER, add_special_tokens=False),
                     "mt_ids": list(r.mt_ids),
                     "ref_ids": T.tokenise(str(s.reference), TOKENISER, add_special_tokens=False)})
        meta.append({"bucket": r.bucket, "mqm_score": s.mqm_score})
    meta = pd.DataFrame(meta)
    print(f"[eval] {val_lang}: {len(nat_recs)} canonical, {len(recs)} re-segmentation", flush=True)

    out = []
    for lam in LAMBDAS:
        set_lambda(model, lam)
        with torch.no_grad():
            nat_s = score_eval_batch(model, nat_recs, device, batch_size=BATCH)
            reseg_s = score_eval_batch(model, recs, device, batch_size=BATCH)
        rho_canon = sps.spearmanr(nat_s, nat_mqm)[0]
        m = meta.copy(); m["score"] = reseg_s
        per_bucket = {}
        for b in MAIN_BUCKETS:
            sb = m[m.bucket == b]
            per_bucket[b] = (sps.spearmanr(sb.score, sb.mqm_score)[0]
                             if len(sb) > 2 and sb.score.nunique() > 1 else float("nan"))
        out.append({"lambda": lam, "rho_canonical": rho_canon,
                    "rho_most_fragmented": per_bucket[MAIN_BUCKETS[-1]],
                    "rho_mildest": per_bucket[MAIN_BUCKETS[0]],
                    "base_rho_canonical_stored": base_rho_stored,
                    "n_canonical": len(nat_recs), "n_e1": len(recs),
                    **{f"rho_{b}": v for b, v in per_bucket.items()}})
        print(f"   lambda={lam:.2f} canonical={rho_canon:.4f} "
              f"most_frag={per_bucket[MAIN_BUCKETS[-1]]:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", required=True, choices=LANGS, help="outer held-out language X")
    ap.add_argument("--inner", required=True, choices=LANGS, help="inner validation language V")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_sd", type=float, default=0.5)
    ap.add_argument("--lambda_anch", type=float, default=1.0)
    args = ap.parse_args()
    assert args.outer != args.inner, "inner validation language must differ from the outer held-out language"

    tag = f"{args.outer}__{args.inner}"
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg["lang"] = seg["lang"].astype(str); seg["condition"] = seg["condition"].astype(str)
    e1 = pd.read_parquet(C.DATA_INTERIM / "reseg_samples.parquet")

    train_langs = [l for l in LANGS if l not in (args.outer, args.inner)]
    train_df = build_training_examples(e1, seg, train_langs,
                                       [args.outer, args.inner], np.random.RandomState(SEED))

    model = MP.load_comet("comet22", device=device)
    replaced = L.inject_lora(model.encoder, target_names=("query", "key", "value", "dense"),
                             r=R, alpha=ALPHA, path_filter=is_attention_linear)
    assert len(replaced) == 96, f"expected 96, got {len(replaced)}"
    for n, p in model.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)
    model.to(device)

    log, mu, sigma, step = train_anchored(model, train_df, device, tag, args.epochs,
                                          args.batch_size, args.lr,
                                          args.lambda_sd, args.lambda_anch)
    pd.DataFrame(log).to_csv(OUT / f"loss_curve_nested_{tag}.csv", index=False)

    model.eval()
    rows = evaluate_on(model, args.inner, seg, e1, device)
    df = pd.DataFrame(rows)
    df.insert(0, "inner_val", args.inner)
    df.insert(0, "outer_held_out", args.outer)
    df["train_langs"] = ",".join(train_langs)
    df["step"] = step
    df.to_csv(OUT / f"nested_frontier_{tag}.csv", index=False)

    with open(OUT / f"DONE_{tag}.json", "w") as f:
        json.dump({"outer": args.outer, "inner": args.inner, "train_langs": train_langs,
                   "n_train_rows": len(train_df), "steps": step,
                   "n_canonical": int(df.n_canonical.iloc[0]),
                   "n_e1": int(df.n_e1.iloc[0])}, f, indent=2)
    print(f"[done] {tag}")


if __name__ == "__main__":
    main()

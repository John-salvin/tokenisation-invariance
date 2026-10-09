#!/usr/bin/env python3
"""Score WMT translations in their native script and romanised, with one
metric per call, for the WMT analyses of Sections 5.2, 5.3, 6 and 7. It uses
the original scoring code (experiments/common/metrics_panel.py), so
scores are computed exactly as the IndicMT Eval panel was. Models are loaded
from a staged copy under $TOKINV_WORK_ROOT/staging (see
experiments/common/constants.py).

Input: one parquet from experiments/data_prep/romanise_wmt.py. Output: the same rows
with `<metric>_native`, `<metric>_romanised`, and XLM-R token counts so items
beyond 512 positions can be flagged (COMET truncates; the analysis excludes
them and reports the count, it never relies on truncated scores).

    python experiments/script/score_wmt.py --input en-hi.parquet --metric comet22 --out DIR [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SB = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
# the scoring code shared with the other cluster scripts: experiments/common/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import metrics_panel as MP  # noqa: E402
import tokenisers as T  # noqa: E402

COMET = {"comet22", "cometkiwi22", "cometkiwi23", "xcomet_xl", "xcomet_xxl"}
CONDS = {"native": ("mt_native", "ref_native"), "romanised": ("mt_rom", "ref_rom")}
# Section 6 conditions (experiments/meaning/wmt_dose_conditions.py), scored with --dose
DOSE_CONDS = {"lossy": ("mt_lossy", "ref_lossy"),
              **{f"noise{x}": (f"mt_noise{x}", f"ref_noise{x}") for x in ("02", "05", "10", "20")}}


def score_comet(name, df, mt, ref, bs):
    model = MP.load_comet(name, device="cuda")
    samples = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(df.src, df[mt], df[ref])]
    return list(model.predict(samples, batch_size=bs, gpus=1, progress_bar=False).scores)


def score_metricx(df, mt, ref, bs=16):
    """Batched, on GPU. Same template and truncation (1024) as the original
    metricx_score_from_text; its 'drop the final EOS' is done by masking the
    EOS position of each row. Checked against the original by --validate."""
    import torch
    model, tok = MP.load_metricx24()
    model = model.to("cuda").eval()
    texts = [MP._metricx_input_text(s, m, r) for s, m, r in zip(df.src, df[mt], df[ref])]
    out = []
    for b in range(0, len(texts), bs):
        enc = tok(texts[b:b + bs], return_tensors="pt", truncation=True, max_length=1024, padding=True)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        last = mask.sum(1) - 1                       # the EOS of each row
        mask[torch.arange(len(last)), last] = 0
        ids[torch.arange(len(last)), last] = tok.pad_token_id
        with torch.no_grad():
            pred = model.forward(ids.to("cuda"), mask.to("cuda"))
        out += [float(x) for x in pred.flatten().cpu()]
    return out


def score_bleurt(df, mt, ref, bs=32):
    """Batched, on GPU, truncated to BLEURT-20's 512 positions (the original
    did not truncate; items beyond 512 are flagged by the token counts)."""
    import torch
    model, tok = MP.load_bleurt20()
    model = model.to("cuda").eval()
    refs, cands = list(df[ref]), list(df[mt])
    out = []
    for b in range(0, len(refs), bs):
        enc = tok(refs[b:b + bs], cands[b:b + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=512)
        with torch.no_grad():
            out += [float(x) for x in model(**{k: v.to("cuda") for k, v in enc.items()}).logits.flatten().cpu()]
    return out


def validate(df, metric, out: Path, n=30):
    """Score n items with the original single-item functions and with the
    batched ones; write the largest absolute difference per condition to a
    JSON file (cluster logs can lose their tail). For BLEURT, only items that
    fit its 512 positions are compared: the original does not truncate."""
    # both functions share one cached model; run every original (CPU) call
    # before any batched call moves that model to the GPU
    sel, orig = {}, {}
    for cond, (mt, ref) in CONDS.items():
        d = df
        if metric == "bleurt20":
            _, tok = MP.load_bleurt20()
            fits = [len(tok(r, m)["input_ids"]) <= 512 for m, r in zip(df[mt], df[ref])]
            d = df[fits]
        sel[cond] = d.head(n)
        d = sel[cond]
        orig[cond] = ([MP.metricx_score_from_text(s, m, r) for s, m, r in zip(d.src, d[mt], d[ref])]
                      if metric == "metricx24" else
                      [MP.bleurt_score_from_text(r, m) for m, r in zip(d[mt], d[ref])])
    res = {}
    for cond, (mt, ref) in CONDS.items():
        d = sel[cond]
        new = score_metricx(d, mt, ref) if metric == "metricx24" else score_bleurt(d, mt, ref)
        res[cond] = {"n": len(d), "max_abs_diff": float(np.max(np.abs(np.array(orig[cond]) - np.array(new))))}
    (out / f"validate_{metric}.json").write_text(json.dumps(res))
    print("VALIDATE", metric, res, flush=True)


MBERT = SB / "staging" / "models" / "mbert_full"   # bert-base-multilingual-cased


def score_bertscore(df, mt, ref, bs=64):
    """F1 of bert_score with the settings behind IndicMT Eval's BERTScore column:
    bert_score.score(lang=<code>) selects bert-base-multilingual-cased, layer 9,
    no idf, no baseline rescaling. Given as a local path, so the layer is explicit."""
    import bert_score
    import torch
    _, _, f1 = bert_score.score(list(df[mt]), list(df[ref]), model_type=str(MBERT), num_layers=9,
                                idf=False, rescale_with_baseline=False, batch_size=bs,
                                device="cuda" if torch.cuda.is_available() else "cpu")
    return [float(x) for x in f1]


def g2_tokenisers():
    """The tokenisers MetricX-24, BLEURT-20 and BERTScore truncate with, loaded
    as metrics_panel loads them but without the models."""
    import constants as C
    from transformers import AutoTokenizer
    sys.path.insert(0, str(C.STAGING / "repos"))
    from bleurt_pytorch.bleurt.tokenization_bleurt_sp import BleurtSPTokenizer
    return (AutoTokenizer.from_pretrained(str(C.STAGING / "models" / "mt5")),
            BleurtSPTokenizer.from_pretrained(str(C.STAGING / "models" / "bleurt20")),
            AutoTokenizer.from_pretrained(str(MBERT)))


def score_surface(df, mt, ref):
    import sacrebleu
    bleu = [sacrebleu.sentence_bleu(m, [r]).score for m, r in zip(df[mt], df[ref])]
    chrf = [sacrebleu.sentence_chrf(m, [r]).score for m, r in zip(df[mt], df[ref])]
    ter = [sacrebleu.sentence_ter(m, [r]).score for m, r in zip(df[mt], df[ref])]
    return {"bleu": bleu, "chrf": chrf, "ter": ter}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--metric", required=True,
                    choices=sorted(COMET | {"metricx24", "bleurt20", "bertscore", "surface", "tokens", "tokens_g2"}))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dose", action="store_true", help="score the Section 6 conditions instead")
    ap.add_argument("--validate", action="store_true",
                    help="compare batched MetricX/BLEURT with the original single-item code")
    a = ap.parse_args()
    df = pd.read_parquet(a.input)
    df = df[df.mt_native_chars > 0].reset_index(drop=True)   # canary lines carry no signal
    if a.limit:
        df = df.head(a.limit)
    a.out.mkdir(parents=True, exist_ok=True)
    pair = a.input.stem
    if a.validate:
        validate(df, a.metric, a.out)
        return
    res = df[["system", "seg_idx", "human", "roundtrip_ok"]].copy()
    if a.dose:
        res = res.join(df[[c for c in df.columns if c.startswith("cer_")]])
    timing = {}
    toks = g2_tokenisers() if a.metric == "tokens_g2" else None
    for cond, (mt, ref) in (DOSE_CONDS if a.dose else CONDS).items():
        t0 = time.time()
        if a.metric in COMET:
            res[f"{a.metric}_{cond}"] = score_comet(a.metric, df, mt, ref, a.batch_size)
        elif a.metric == "metricx24":
            res[f"metricx24_{cond}"] = score_metricx(df, mt, ref)
        elif a.metric == "bleurt20":
            res[f"bleurt20_{cond}"] = score_bleurt(df, mt, ref)
        elif a.metric == "bertscore":
            res[f"bertscore_{cond}"] = score_bertscore(df, mt, ref)
        elif a.metric == "surface":
            for k, v in score_surface(df, mt, ref).items():
                res[f"{k}_{cond}"] = v
        elif a.metric == "tokens":
            n = lambda s: len(T.tokenise(str(s), "xlmr", add_special_tokens=False))
            res[f"ntok_mt_{cond}"] = [n(x) for x in df[mt]]
            res[f"ntok_ref_{cond}"] = [n(x) for x in df[ref]]
            if cond == "native":
                res["ntok_src"] = [n(x) for x in df.src]
        elif a.metric == "tokens_g2":
            # full lengths, special tokens included, to compare with each
            # metric's limit: MetricX 1024, BLEURT 512 (pair), mBERT 512 (each side)
            mt5, blt, mb = toks
            res[f"ntok_metricx_{cond}"] = [len(mt5(MP._metricx_input_text(s, m, r))["input_ids"])
                                           for s, m, r in zip(df.src, df[mt], df[ref])]
            res[f"ntok_bleurt_{cond}"] = [len(blt(r, m)["input_ids"]) for m, r in zip(df[mt], df[ref])]
            res[f"ntok_mbert_mt_{cond}"] = [len(mb(m)["input_ids"]) for m in df[mt]]
            res[f"ntok_mbert_ref_{cond}"] = [len(mb(r)["input_ids"]) for r in df[ref]]
        timing[cond] = {"seconds": round(time.time() - t0, 1), "items": len(df),
                        "items_per_s": round(len(df) / max(time.time() - t0, 1e-9), 2)}
        print(f"{pair} {a.metric} {cond}: {timing[cond]}", flush=True)
    tag = f"{pair}_{a.metric}" + ("_dose" if a.dose else "") + (f"_pilot{a.limit}" if a.limit else "")
    res.to_parquet(a.out / f"{tag}.parquet", index=False)
    (a.out / f"{tag}_timing.json").write_text(json.dumps(timing))
    print(f"wrote {tag}", flush=True)


if __name__ == "__main__":
    main()

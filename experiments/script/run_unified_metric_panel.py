#!/usr/bin/env python3
"""CometKiwi-22/23 and xCOMET-XL/XXL scores on native and romanised text.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import metrics_panel as MP
import provenance as PROV
import tokenisers as T
from scipy import stats as sps

TOKENISER = "xlmr"
BATCH_SIZE = 16
MODELS_IDS_PATH = ["cometkiwi22", "cometkiwi23"]
MODELS_TEXT_PATH = ["xcomet_xl", "xcomet_xxl"]
MODELS_ASCENDING_COST = MODELS_IDS_PATH + MODELS_TEXT_PATH
N_PER_LANG_FULL = None
N_PER_LANG_XL = 300
N_PER_LANG_XXL = 300


def check_gate_passed(name: str) -> None:
    gate_path = C.OUT_TABLES / "m4_unified_equivalence_check.csv"
    if not gate_path.exists():
        raise MP.MetricsPanelError(f"equivalence gate not run yet -- run "
                                   f"run_m4_unified_verify.py before scoring {name}")
    gate = pd.read_csv(gate_path)
    sub = gate[gate.model == name]
    if len(sub) == 0:
        raise MP.MetricsPanelError(f"{name} not present in the equivalence gate results -- "
                                   f"run run_m4_unified_verify.py through this model first")
    if (sub.abs_diff > 1e-6).any():
        raise MP.MetricsPanelError(f"STOP: {name} did NOT pass the equivalence gate "
                                   f"(max diff {sub.abs_diff.max():.2e}). Not scoring.")


def score_model_ids(name: str, seg: pd.DataFrame, device: str, n_per_lang: int | None) -> pd.DataFrame:
    check_gate_passed(name)
    model = MP.load_comet(name, device=device)
    referenceless = MP._is_referenceless(model)
    print(f"{name}: referenceless={referenceless}, n_per_lang={n_per_lang or 'FULL'} (ids path)")

    rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond) & seg.mqm_valid]
            if n_per_lang is not None and len(sub) > n_per_lang:
                sub = sub.sample(n=n_per_lang, random_state=C.SEED)
            for start in range(0, len(sub), BATCH_SIZE):
                batch = sub.iloc[start:start + BATCH_SIZE]
                src_b = [T.tokenise(str(t), TOKENISER, add_special_tokens=False) for t in batch.source_en]
                mt_b = [T.tokenise(str(t), TOKENISER, add_special_tokens=False) for t in batch.target]
                ref_b = (None if referenceless else
                        [T.tokenise(str(t), TOKENISER, add_special_tokens=False) for t in batch.reference])
                scores = MP.score_from_ids_batch(model, src_b, mt_b, ref_b)
                for (_, r), s in zip(batch.iterrows(), scores):
                    rows.append({"model": name, "lang": lang, "condition": cond,
                                "segment_id": r.segment_id, "score": s, "mqm_score": r.mqm_score})
            print(f"  {lang} {cond}: {len(sub)} scored")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_TABLES / f"unified_metric_{name}_scores.csv", index=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return df


def score_model_text(name: str, seg: pd.DataFrame, device: str, n_per_lang: int | None) -> pd.DataFrame:
    model = MP.load_comet(name, device=device)
    model = model.to(torch.bfloat16).to(device)
    print(f"{name}: n_per_lang={n_per_lang or 'FULL'} (score_from_text path, per-segment)")

    rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = seg[(seg.lang == lang) & (seg.condition == cond) & seg.mqm_valid]
            if n_per_lang is not None and len(sub) > n_per_lang:
                sub = sub.sample(n=n_per_lang, random_state=C.SEED)
            for _, r in sub.iterrows():
                s = MP.score_from_text(model, str(r.source_en), str(r.target), str(r.reference))
                rows.append({"model": name, "lang": lang, "condition": cond,
                            "segment_id": r.segment_id, "score": s, "mqm_score": r.mqm_score})
            print(f"  {lang} {cond}: {len(sub)} scored")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_TABLES / f"unified_metric_{name}_scores.csv", index=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return df


def summarize(name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang in C.LANG_ORDER:
        for cond in C.CONDITIONS:
            sub = df[(df.lang == lang) & (df.condition == cond)]
            rho = float(sps.spearmanr(sub["score"], sub["mqm_score"])[0])
            rows.append({"lang": lang, "condition": cond, "n": len(sub),
                        f"mean_{name}": float(sub["score"].mean()), "spearman_vs_mqm": rho})
    return pd.DataFrame(rows)


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    models_to_run = sys.argv[1:] if len(sys.argv) > 1 else MODELS_ASCENDING_COST
    unknown = [m for m in models_to_run if m not in MODELS_ASCENDING_COST]
    if unknown:
        raise MP.MetricsPanelError(f"unknown model(s) in argv: {unknown}")

    summaries = {}
    for name in models_to_run:
        if name == "xcomet_xxl":
            n = N_PER_LANG_XXL
        elif name == "xcomet_xl":
            n = N_PER_LANG_XL
        else:
            n = N_PER_LANG_FULL
        print("\n" + "=" * 78); print(name); print("=" * 78)
        try:
            score_fn = score_model_text if name in MODELS_TEXT_PATH else score_model_ids
            df = score_fn(name, seg, device, n)
        except Exception as exc:
            print(f"!!! {name} FAILED, not scored, continuing to report what's already done: {exc}")
            continue
        summary = summarize(name, df)
        summary.to_csv(C.OUT_TABLES / f"unified_metric_{name}_summary.csv", index=False)
        summaries[name] = summary
        print(summary.to_string(index=False))

        PROV.register(
            f"unified_metric_{name}", {
                "summary": summary.to_dict("records"), "n_per_lang": n or "full_corpus",
            },
            PROV.stamp(Path(__file__), modules=["metrics_panel", "tokenisers", "constants"],
                       seed=C.SEED, note=f"{name} native/romanised panel"))

    print("\n" + "=" * 78); print("Combined panel (native/romanised, all models scored so far)")
    print("=" * 78)
    if summaries:
        new_combined = pd.concat(
            [s.assign(model=name) for name, s in summaries.items()], ignore_index=True)
        combined_path = C.OUT_TABLES / "unified_metric_panel_combined.csv"
        if combined_path.exists():
            prior = pd.read_csv(combined_path)
            prior = prior[~prior.model.isin(summaries.keys())]
            combined = pd.concat([prior, new_combined], ignore_index=True)
        else:
            combined = new_combined
        combined.to_csv(C.OUT_TABLES / "unified_metric_panel_combined.csv", index=False)
        print(combined.to_string(index=False))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

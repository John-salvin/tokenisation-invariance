#!/usr/bin/env python3
"""Rebuild every table of the paper from the committed data, without
notebooks (the notebooks in notebooks/ do the same, step by step).

    python scripts/run_all.py            # everything (about 25 minutes)
    python scripts/run_all.py --only qn  # one stage
    python scripts/run_all.py --list

Each stage is one module of src/tokinv and corresponds to a part of the paper.
Tables are written to results/tables/. No GPU is used and no model weights
are downloaded; the tokeniser stages fetch vocabulary files (a few MB) from
the Hugging Face Hub at pinned revisions on first use.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# (stage, module, what it covers), in paper order. Every stage is independent
# of the others, so any subset can be run.
STAGES = [
    ("resegmentation", "tokinv.resegmentation", "Sec 4.1-4.2, Tables 1-2, App. Per-Bucket Results: re-segmentation, MetricX replication"),
    ("equal_count", "tokinv.equal_count", "Sec 4.2, App. Near-Canonical Token Counts"),
    ("layerwise", "tokinv.layerwise", "Sec 4.3, App. Layerwise Retrieval Probe"),
    ("metaeval", "tokinv.metaeval", "Sec 5.1, 5.3, Table 3, App. Meta-Evaluation Table"),
    ("capacity", "tokinv.metaeval:run_capacity", "App. Capacity and Degradation"),
    ("qn", "tokinv.qn", "Sec 7, App. Quantile Normalisation"),
    ("repair", "tokinv.repair", "Sec 8, App. Selecting lambda, Hyperparameter Frontier (incl. nested mu), Zero-Shot Transfer"),
    ("adequacy", "tokinv.adequacy", "Sec 6, App. Distortion, Token Parity by Scheme"),
    ("beyond_indic", "tokinv.beyond_indic", "Sec 5.2-5.3, 6, Table 4: Script Invariance on WMT (7 settings, 11 metrics)"),
    ("serbian", "tokinv.serbian", "Sec 5.2, App. Serbian"),
    ("effect_predictors", "tokinv.effect_predictors", "Limitations, App. Predictors of Effect Size, Vocabulary ceiling"),
    ("tokeniser_stats", "tokinv.tokeniser_stats", "Sec 4.2, App. Vocabulary, Sampler, Romanisation Across Tokenisers"),
    ("data_quality", "tokinv.data_quality", "App. Data Quality Findings and worked examples"),
]


def run_stage(target: str) -> None:
    mod, _, fn = target.partition(":")
    getattr(importlib.import_module(mod), fn or "run")()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="stage names to run")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for name, _, what in STAGES:
            print(f"{name:16s} {what}")
        return 0
    chosen = [s for s in STAGES if not a.only or s[0] in a.only]
    if a.only and len(chosen) != len(a.only):
        unknown = set(a.only) - {s[0] for s in STAGES}
        print(f"unknown stage(s): {sorted(unknown)}; see --list")
        return 2
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    log = []
    t_all = time.time()
    for name, target, what in chosen:
        t0 = time.time()
        print(f"--- {name:16s} {what}", flush=True)
        run_stage(target)
        dt = time.time() - t0
        log.append(f"| {name} | {dt:7.1f} s |")
        print(f"    done in {dt:.1f} s", flush=True)
    total = time.time() - t_all
    if not a.only:
        (ROOT / "results" / "logs" / "pipeline_timing.md").write_text(
            "| stage | time |\n|---|---|\n" + "\n".join(log) + f"\n| **total** | {total:7.1f} s |\n")
    print(f"\nAll {len(chosen)} stage(s) finished in {total / 60:.1f} min. Tables in results/tables/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

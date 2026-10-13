"""Section 4.3 and App. Layerwise Retrieval Probe.

At each XLM-R-large layer, retrieve the English source from the target's
mean-pooled representation (accuracy@1 over 200 segments per language), for
native, romanised and re-segmented targets. The re-segmented condition uses
the [1.50,1.75) bucket at seed 0: clearly non-canonical, far from the extreme.

Inputs (experiments/tokenisation/run_layerwise.py and
run_layerwise_bootstrap.py, GPU):
  layerwise_acc1.csv       the seed-0 probe, every layer
  layerwise_bootstrap.csv  three seed replicates plus bucket replicates,
                              each with its own BCa interval
"""
from __future__ import annotations

import pandas as pd

from .paths import DATA, TABLES

L = DATA / "derived" / "layerwise"


def seed_summary() -> pd.DataFrame:
    df = pd.read_csv(L / "layerwise_bootstrap.csv")
    return (df[df.replicate_type == "seed"].groupby(["lang", "condition", "layer"]).acc1
              .agg(["mean", "std", "min", "max", "count"]).reset_index()
              .rename(columns={"mean": "seed_mean_acc1", "std": "seed_sd_acc1",
                               "min": "seed_min_acc1", "max": "seed_max_acc1",
                               "count": "n_seeds"}))


def peaks(summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lang, cond), g in summ.groupby(["lang", "condition"]):
        p = g.loc[g.seed_mean_acc1.idxmax()]
        rows.append({"lang": lang, "condition": cond, "peak_layer": int(p.layer),
                     "peak_seed_mean_acc1": float(p.seed_mean_acc1),
                     "peak_seed_sd_acc1": float(p.seed_sd_acc1)})
    return pd.DataFrame(rows)


def run() -> None:
    pd.read_csv(L / "layerwise_acc1.csv").to_csv(TABLES / "layerwise_acc1.csv", index=False)
    s = seed_summary()
    s.to_csv(TABLES / "layerwise_seed_summary.csv", index=False)
    peaks(s).to_csv(TABLES / "layerwise_peak_with_uncertainty.csv", index=False)


if __name__ == "__main__":
    run()

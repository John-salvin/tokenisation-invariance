"""Nested selection of lambda from the inner-fold frontiers (App. Selecting lambda).

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

SB_ROOT = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = Path(os.environ.get("ROB_ROOT", SB_ROOT / "repo" / "cluster_copy"))
TAB = ROB / "results" / "tables"
NESTED = TAB / "nested"

LANGS = ["guj", "tam", "mal", "mar", "hin"]
LAMBDAS = [0.25, 0.5, 0.75, 1.0]
COST_CAP = 0.02


def frontier_stats(df: pd.DataFrame) -> dict:
    g = df.set_index("lambda")
    f0, c0 = g.loc[0.0, "rho_most_fragmented"], g.loc[0.0, "rho_canonical"]
    return {lam: {"gain": g.loc[lam, "rho_most_fragmented"] - f0,
                  "cost": c0 - g.loc[lam, "rho_canonical"]} for lam in LAMBDAS}


def main():
    inner_rows = []
    for outer in LANGS:
        for inner in LANGS:
            if inner == outer:
                continue
            f = NESTED / f"nested_frontier_{outer}__{inner}.csv"
            if not f.exists():
                raise FileNotFoundError(f)
            st = frontier_stats(pd.read_csv(f))
            for lam in LAMBDAS:
                inner_rows.append({"outer_held_out": outer, "inner_val": inner,
                                   "lambda": lam, **st[lam]})
    inner = pd.DataFrame(inner_rows)
    inner.to_csv(TAB / "nested_lambda_inner.csv", index=False)

    agg = (inner.groupby(["outer_held_out", "lambda"])
                .agg(mean_gain=("gain", "mean"), mean_cost=("cost", "mean"),
                     worst_cost=("cost", "max"), n_inner=("gain", "size"))
                .reset_index())
    agg.to_csv(TAB / "nested_lambda_validation.csv", index=False)

    outer_plain = frontier_stats_by_fold(TAB / "lora_anchored_frontier.csv")
    outer_anch = frontier_stats_by_fold(TAB / "lora_anchored_frontier_anchored.csv")

    sel = []
    for outer in LANGS:
        g = agg[agg.outer_held_out == outer].set_index("lambda")
        feasible = g[g.mean_cost <= COST_CAP]
        if len(feasible):
            best = feasible.sort_values(
                ["mean_gain", "mean_cost"], ascending=[False, True]).index[0]
            infeasible = False
        else:
            best = g.mean_cost.idxmin()
            infeasible = True
        held = outer_anch[outer][best]
        published = outer_plain[outer][1.0]["gain"]
        sel.append({
            "held_out": outer, "lambda_star": best,
            "constraint_infeasible": infeasible,
            "val_mean_gain": g.loc[best, "mean_gain"],
            "val_mean_cost": g.loc[best, "mean_cost"],
            "heldout_gain": held["gain"],
            "heldout_cost": held["cost"],
            "published_repair": published,
            "heldout_retention_pct": 100 * held["gain"] / published,
        })
    s = pd.DataFrame(sel)
    s.to_csv(TAB / "nested_lambda_selection.csv", index=False)

    print("== inner-validation means, per outer fold ==")
    print(agg.round(4).to_string(index=False))
    print("\n== selection and held-out result ==")
    print(s.round(4).to_string(index=False))
    print("\n== summary ==")
    print(f"lambda* values: {dict(zip(s.held_out, s.lambda_star))}")
    print(f"folds selecting 0.5: {(s.lambda_star == 0.5).sum()} / 5")
    print(f"median held-out retention: {s.heldout_retention_pct.median():.1f}%")
    print(f"min held-out retention:    {s.heldout_retention_pct.min():.1f}%")
    print(f"mean held-out canonical cost: {s.heldout_cost.mean():+.4f}")
    print(f"worst held-out canonical cost: {s.heldout_cost.max():+.4f}")


def frontier_stats_by_fold(path: Path) -> dict:
    df = pd.read_csv(path)
    return {h: frontier_stats(g) for h, g in df.groupby("held_out")}


if __name__ == "__main__":
    main()

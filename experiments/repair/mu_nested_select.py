"""Nested selection of (mu_sd, mu_an, lambda), by the same rule as the
selection of lambda (App. Selecting lambda). Inner statistics use
nested_lambda_select.frontier_stats, so gain and cost are defined exactly as for
the lambda selection. Held-out numbers are read only after the selection.
Writes mu_nested_inner.csv, mu_nested_selection.csv, mu_nested_sensitivity.csv."""
import os, sys
from pathlib import Path
import pandas as pd
SB = Path(os.environ.get("TOKINV_WORK_ROOT", Path.home() / "tokinv_work"))
ROB = SB / "repo" / "cluster_copy"
# nested_lambda_select.py sits next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nested_lambda_select import frontier_stats, LANGS, LAMBDAS, COST_CAP  # noqa: E402
TAB = ROB / "results" / "tables"
OUT = SB / "mu_nested" / "results"
OUT.mkdir(parents=True, exist_ok=True)
GRID = [(sd, an) for sd in (0.0, 0.5, 1.0) for an in (0.5, 1.0, 2.0)]
REPORTED = (0.5, 1.0)


def inner_dir(sd, an):
    return TAB / "nested" if (sd, an) == REPORTED else SB / "wmt" / "mu" / "nested" / f"sd{sd}_an{an}"


rows = []
for sd, an in GRID:
    for X in LANGS:
        for V in LANGS:
            if V == X:
                continue
            f = inner_dir(sd, an) / f"nested_frontier_{X}__{V}.csv"
            if not f.exists():
                sys.exit(f"missing {f}")
            st = frontier_stats(pd.read_csv(f))
            for lam in LAMBDAS:
                rows.append({"mu_sd": sd, "mu_an": an, "outer": X, "inner": V, "lambda": lam, **st[lam]})
inner = pd.DataFrame(rows)
inner.to_csv(OUT / "mu_nested_inner.csv", index=False)
agg = inner.groupby(["outer", "mu_sd", "mu_an", "lambda"]).agg(
    mean_gain=("gain", "mean"), mean_cost=("cost", "mean")).reset_index()

sel = []
for X in LANGS:
    cands = []
    for sd, an in GRID:
        g = agg[(agg.outer == X) & (agg.mu_sd == sd) & (agg.mu_an == an)].set_index("lambda")
        feas = g[g.mean_cost <= COST_CAP]
        if len(feas):  # the lambda rule within configuration c
            lam = feas.sort_values(["mean_gain", "mean_cost"], ascending=[False, True]).index[0]
            cands.append({"mu_sd": sd, "mu_an": an, "lambda": lam, "gain": g.loc[lam, "mean_gain"],
                          "cost": g.loc[lam, "mean_cost"], "reported": (sd, an) == REPORTED})
    c = pd.DataFrame(cands)
    # tie-breaks: larger gain, smaller cost, reported config, smaller mu_an, smaller mu_sd
    c = c.sort_values(["gain", "cost", "reported", "mu_an", "mu_sd"], ascending=[False, True, False, True, True])
    best = c.iloc[0]
    sel.append({"outer": X, "n_feasible_configs": len(c), "mu_sd": best.mu_sd, "mu_an": best.mu_an,
                "lambda": best["lambda"], "inner_gain": best.gain, "inner_cost": best.cost})
s = pd.DataFrame(sel)


def outer_frontier(sd, an):
    """Held-out frontier of the outer adapters of a configuration."""
    f = TAB / f"lora_anchored_frontier_r8_sd{sd}_an{an}.csv"
    return {h: frontier_stats(g) for h, g in pd.read_csv(f).groupby("held_out")}


rep = {h: frontier_stats(g) for h, g in pd.read_csv(TAB / "lora_anchored_frontier_anchored.csv").groupby("held_out")}
repsel = pd.read_csv(TAB / "nested_lambda_selection.csv").set_index("held_out")
out = []
for r in s.to_dict("records"):
    held = outer_frontier(r["mu_sd"], r["mu_an"])[r["outer"]][r["lambda"]]
    lr = repsel.loc[r["outer"], "lambda_star"]
    out.append({**r, "heldout_gain": held["gain"], "heldout_cost": held["cost"],
                "reported_lambda": lr, "reported_heldout_gain": rep[r["outer"]][lr]["gain"],
                "reported_heldout_cost": rep[r["outer"]][lr]["cost"],
                # like-for-like: the grid's own (0.5, 1) adapter, evaluated as c* is
                "grid_reported_heldout_gain": outer_frontier(*REPORTED)[r["outer"]][lr]["gain"],
                "grid_reported_heldout_cost": outer_frontier(*REPORTED)[r["outer"]][lr]["cost"]})
pd.DataFrame(out).to_csv(OUT / "mu_nested_selection.csv", index=False)

# plain sensitivity check (no selection): mu_an in {0.5, 2} at mu_sd = 0.5, held out
sens = []
for an in (0.5, 1.0, 2.0):
    fr = outer_frontier(0.5, an)
    for X in LANGS:
        for lam in LAMBDAS:
            sens.append({"mu_sd": 0.5, "mu_an": an, "held_out": X, "lambda": lam, **fr[X][lam]})
pd.DataFrame(sens).to_csv(OUT / "mu_nested_sensitivity.csv", index=False)
print(pd.read_csv(OUT / "mu_nested_selection.csv").round(4).to_string(index=False))

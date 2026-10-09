"""Section 8 and App. Selecting lambda, Hyperparameter Frontier, Zero-Shot
Transfer: repairing the sensitivity fault.

Inputs are the evaluations of the adapters, trained and evaluated on GPUs
(experiments/repair/) and committed under ``data/derived/repair/``:

  lolo_plain/          per held-out language: the plain multi-tokenisation
                       adapter's canonical regression and dose response
  frontier_plain.csv   per held-out language x lambda: agreement on canonical
  frontier_anchored.csv  text and in every bucket, plain and anchored adapters
  frontier_grid/       the same for LoRA rank r x lambda_sd (12 configurations)
  nested/              20 inner-fold frontiers for nested lambda selection
  mu_nested/           nested selection of (mu_sd, mu_an, lambda): the cluster's
                       inner statistics and per-fold choice, plus frontier/ with
                       the rank-8 outer frontiers at mu_an in {0.5, 2}
  transfer/            per-segment WMT scores: base COMET-22 and the Indic
                       adapter applied zero-shot at several lambdas

lambda interpolates the adapter: 0 is base COMET-22, 1 the full adapter.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import offtarget, stats
from .paths import DATA, INDIC, INDIC_LANGS, TABLES

R = DATA / "derived" / "repair"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
COST_CAP = 0.02          # the lambda rule: mean inner canonical cost <= 0.02
RETAIN_MIN = 0.70        # Pareto: keep >= 70% of the full-adapter gain ...
LOSS_MAX = 0.02          # ... at <= 0.02 canonical loss


# ------------------------------------- Section 8, A1 (plain, LOLO) -- #
def lolo_plain() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cat = lambda k: pd.concat([pd.read_csv(R / "lolo_plain" / f"{k}_{l}.csv")
                               for l in INDIC_LANGS], ignore_index=True)
    return cat("dose_response"), cat("canonical_regression"), cat("native_romanised")


# ------------------------- Table 8 and App. Hyperparameter Frontier -- #
def frontier_analysis(frontier: pd.DataFrame) -> pd.DataFrame:
    """Gain in the most fragmented bucket and loss on canonical text, relative
    to lambda = 0 (base COMET-22), per held-out language."""
    rows = []
    for held, g in frontier.groupby("held_out"):
        g = g.set_index("lambda")
        c0, f0 = g.loc[0.0, "rho_canonical"], g.loc[0.0, "rho_most_fragmented"]
        full = g.loc[1.0, "rho_most_fragmented"] - f0
        for lam in LAMBDAS:
            gain = g.loc[lam, "rho_most_fragmented"] - f0
            loss = c0 - g.loc[lam, "rho_canonical"]
            rows.append({"held_out": held, "lambda": lam, "frag_gain": gain,
                         "frag_gain_retained_pct": 100 * gain / full if full else float("nan"),
                         "canonical_loss": loss,
                         "pareto_ok": bool(full and gain / full >= RETAIN_MIN and loss <= LOSS_MAX)})
    return pd.DataFrame(rows)


def frontier_grid(lam: float) -> pd.DataFrame:
    """App. Hyperparameter Frontier: is the repair a lucky hyperparameter? Every (r, lambda_sd)."""
    pat = re.compile(r"r(\d+)_sd([\d.]+)_an1\.0\.csv$")
    rows = []
    for f in sorted((R / "frontier_grid").glob("*.csv")):
        r, sd = pat.search(f.name).groups()
        g = frontier_analysis(pd.read_csv(f))
        g = g[g["lambda"] == lam]
        rows.append({"r": int(r), "lambda_sd": float(sd), "folds": len(g),
                     "mean_frag_gain": g.frag_gain.mean(), "min_frag_gain": g.frag_gain.min(),
                     "mean_canonical_loss": g.canonical_loss.mean(),
                     "max_canonical_loss": g.canonical_loss.max(),
                     "pareto_ok_folds": int(g.pareto_ok.sum())})
    return pd.DataFrame(rows).sort_values(["r", "lambda_sd"]).reset_index(drop=True)


def _gain_cost(frontier: pd.DataFrame, key: str = "held_out") -> dict:
    out = {}
    for held, g in frontier.groupby(key):
        g = g.set_index("lambda")
        f0, c0 = g.loc[0.0, "rho_most_fragmented"], g.loc[0.0, "rho_canonical"]
        out[held] = {lam: {"gain": g.loc[lam, "rho_most_fragmented"] - f0,
                           "cost": c0 - g.loc[lam, "rho_canonical"]} for lam in LAMBDAS[1:]}
    return out


def nested_selection() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """App. Selecting lambda: choose lambda on inner folds only (the largest mean
    repair whose mean cost is at most 0.02), then read the held-out
    result off the outer frontier. Nothing here can tune on held-out data."""
    inner_rows = []
    for outer in INDIC_LANGS:
        for inner in INDIC_LANGS:
            if inner == outer:
                continue
            st = _gain_cost(pd.read_csv(R / "nested" / f"nested_frontier_{outer}__{inner}.csv"),
                            key="inner_val")[inner]
            for lam in LAMBDAS[1:]:
                inner_rows.append({"outer_held_out": outer, "inner_val": inner,
                                   "lambda": lam, **st[lam]})
    inner = pd.DataFrame(inner_rows)
    agg = (inner.groupby(["outer_held_out", "lambda"])
                .agg(mean_gain=("gain", "mean"), mean_cost=("cost", "mean"),
                     worst_cost=("cost", "max"), n_inner=("gain", "size")).reset_index())
    plain = _gain_cost(pd.read_csv(R / "frontier_plain.csv"))
    anch = _gain_cost(pd.read_csv(R / "frontier_anchored.csv"))
    sel = []
    for outer in INDIC_LANGS:
        g = agg[agg.outer_held_out == outer].set_index("lambda")
        feasible = g[g.mean_cost <= COST_CAP]
        if len(feasible):
            best = feasible.sort_values(["mean_gain", "mean_cost"], ascending=[False, True]).index[0]
        else:
            best = g.mean_cost.idxmin()
        held, published = anch[outer][best], plain[outer][1.0]["gain"]
        sel.append({"held_out": outer, "lambda_star": best, "cost_cap": COST_CAP,
                    "constraint_infeasible": not len(feasible),
                    "val_mean_gain": g.loc[best, "mean_gain"],
                    "val_mean_cost": g.loc[best, "mean_cost"],
                    "heldout_gain": held["gain"], "heldout_cost": held["cost"],
                    "published_repair": published,
                    "heldout_retention_pct": 100 * held["gain"] / published})
    return inner, agg, pd.DataFrame(sel)


def anchored_vs_plain_retention() -> pd.DataFrame:
    """Anchored adapter's held-out gain at each lambda as a percentage of the
    plain adapter's full (lambda = 1) gain, summarised over the five folds."""
    plain = _gain_cost(pd.read_csv(R / "frontier_plain.csv"))
    anch = _gain_cost(pd.read_csv(R / "frontier_anchored.csv"))
    rows = []
    for lam in LAMBDAS[1:]:
        pct = [100 * anch[h][lam]["gain"] / plain[h][1.0]["gain"] for h in INDIC_LANGS]
        rows.append({"lambda": lam, "median_pct_of_plain": round(float(np.median(pct)), 1),
                     "min_pct": round(float(np.min(pct)), 1),
                     "mean_canonical_cost": round(float(np.mean([anch[h][lam]["cost"] for h in INDIC_LANGS])), 3),
                     "median_gain": round(float(np.median([anch[h][lam]["gain"] for h in INDIC_LANGS])), 3)})
    return pd.DataFrame(rows)


# ----------------------------------- App. Zero-Shot Transfer -------- #
TRANSFER_PAIRS = ["en-ru_RU", "en-ar_EG", "en-cs_CZ", "en-is_IS", "en-uk_UA", "en-hi", "en-cs"]
CROSS_PAIRS = ["en-ru_RU", "en-ar_EG", "en-cs_CZ"]
WMT_BOOT_SEED = 20260929   # seed of the WMT driver's own trend bootstrap


ON_TARGET = True   # main text; run() also writes all-systems variants


def _run(pair: str, tag: str, on_target: bool | None = None) -> pd.DataFrame:
    """One transfer run; off-target systems (tokinv.offtarget) dropped."""
    d = pd.read_parquet(R / "transfer" / f"{pair}_{tag}.parquet")
    keep = ON_TARGET if on_target is None else on_target
    return d[~d.system.isin(offtarget.off_target(pair))] if keep else d


def canonical_and_last(d: pd.DataFrame) -> tuple[float, float]:
    """Agreement on canonical text and in the most fragmented bucket."""
    nc = d[d.bucket != "canonical"]
    last = nc[nc.bucket == sorted(nc.bucket.unique())[-1]]
    can = d[d.bucket == "canonical"]
    return stats.spearman(can.score, can.human), stats.spearman(last.score, last.human)


def zeroshot_repair() -> pd.DataFrame:
    rows = []
    for p in TRANSFER_PAIRS:
        bc, bl = canonical_and_last(_run(p, "base600"))
        ac, al = canonical_and_last(_run(p, "adapt"))
        rows.append({"setting": p, "base_last": round(bl, 3), "adapted_last": round(al, 3),
                     "repair": round(al - bl, 3), "canonical_change": round(ac - bc, 3)})
    return pd.DataFrame(rows)


def transfer_lambda() -> pd.DataFrame:
    rows = []
    for p in [x for x in TRANSFER_PAIRS if x != "en-hi"]:
        p_order = p
        bc, bl = canonical_and_last(_run(p, "base600"))
        r = {"pair": p_order, "base_last": round(bl, 3)}
        for tag, name in [("adapt", "lam1"), ("a3lam0.5", "3ep_lam0.5"), ("a6lam0.5", "6ep_lam0.5")]:
            c, l = canonical_and_last(_run(p, tag))
            r.update({f"{name}_last": round(l, 3), f"{name}_repair": round(l - bl, 3),
                      f"{name}_canon_chg": round(c - bc, 3)})
        rows.append(r)
    order = ["en-ar_EG", "en-ru_RU", "en-cs_CZ", "en-is_IS", "en-uk_UA", "en-cs"]
    return pd.DataFrame(rows).set_index("pair").loc[order].reset_index()


def transfer_lam075() -> pd.DataFrame:
    rows = []
    for p in ["en-ar_EG", "en-ru_RU", "en-cs_CZ", "en-is_IS", "en-uk_UA", "en-cs"]:
        bc, bl = canonical_and_last(_run(p, "base600"))
        rep = lambda tag: round(canonical_and_last(_run(p, tag))[1] - bl, 3)
        c75, l75 = canonical_and_last(_run(p, "b3lam0.75"))
        rows.append({"pair": p, "base_last": round(bl, 3), "rep_lam0.5": rep("a3lam0.5"),
                     "rep_lam1": rep("adapt"), "rep_lam0.75_3ep": round(l75 - bl, 3),
                     "canon_chg_lam0.75": round(c75 - bc, 3), "last_lam0.75": round(l75, 3),
                     "rep_lam0.75_6ep": rep("b6lam0.75"),
                     "last_lam0.75_6ep": round(canonical_and_last(_run(p, "b6lam0.75"))[1], 3)})
    return pd.DataFrame(rows)


def crossscript_fixed() -> pd.DataFrame:
    """Cross-script training (within-language z-normalised labels) against the
    Indic-only adapter applied zero-shot."""
    rows = []
    for p in CROSS_PAIRS:
        base, zs, cf = _run(p, "base600"), _run(p, "adapt"), _run(p, "crossfix")
        bc, bl = canonical_and_last(base)
        _, zl = canonical_and_last(zs)
        cc, cl = canonical_and_last(cf)
        curve = stats.dose_curve(cf)
        mono = stats.spearman(curve.mean_ratio, curve.rho)
        lo, hi = stats.segment_bootstrap(cf.rename(columns={"seg_idx": "unit"}),
                                         stats.monotonicity, "unit", seed=WMT_BOOT_SEED)
        rows.append({"setting": p, "base600_canonical": bc, "base600_last": bl,
                     "zeroshot_last": zl, "zeroshot_repair": zl - bl,
                     "crossfix_canonical": cc, "crossfix_last": cl, "crossfix_repair": cl - bl,
                     "crossfix_canonical_change": cc - bc, "crossfix_monotonicity": mono,
                     "crossfix_exact_p": stats.exact_p(mono, len(curve)),
                     "crossfix_boot_ci_lo": lo, "crossfix_boot_ci_hi": hi,
                     "n_segments": cf.seg_idx.nunique()})
    return pd.DataFrame(rows).round(3)


def source_disjointness() -> pd.DataFrame:
    """No English source the Indic adapter trained on appears in the WMT frame.
    WMT sources are stored as XLM-R ids, so Indic sources are compared as ids."""
    from . import tokenisers as T
    ids = pd.read_csv(R / "indic_train_guj_fold_segments.csv")
    seg = pd.read_parquet(INDIC / "segments.parquet")
    src = (ids.merge(seg[seg.condition == "native"][["segment_id", "source_en"]], on="segment_id")
              .source_en.unique())
    tok = T.get("xlmr")
    indic = {tuple(tok(s, add_special_tokens=False)["input_ids"]) for s in src}
    wmt = {tuple(x) for x in pd.read_parquet(R / "wmt_crossscript_frame_sources.parquet").src_ids}
    return pd.DataFrame({"corpus": ["indic_train_guj_fold", "wmt_eval", "intersection"],
                         "n_unique_sources": [len(indic), len(wmt), len(indic & wmt)]})


def _mu_frontier(sd: float, an: float) -> pd.DataFrame:
    """Held-out frontier of the rank-8 outer adapters at (mu_sd, mu_an)."""
    name = f"lora_anchored_frontier_r8_sd{sd}_an{an}.csv"
    return pd.read_csv(R / "frontier_grid" / name if an == 1.0 else R / "mu_nested" / "frontier" / name)


def mu_nested() -> tuple[pd.DataFrame, pd.DataFrame]:
    """App. Hyperparameter Frontier: the nested selection of (mu_sd, mu_an, lambda)
    (experiments/repair/mu_nested_*), reported as a check on the fixed (0.5, 1).
    The choice per fold is read from the selection record (it needs the 160
    inner adapters); its held-out repair and cost are recomputed here from the
    outer frontiers, next to the grid's own (0.5, 1) adapter at that fold's
    lambda*. Every frontier is evaluated on the same 2,000-row sample."""
    d = R / "mu_nested"
    sel = pd.read_csv(d / "mu_nested_selection.csv")[
        ["outer", "n_feasible_configs", "mu_sd", "mu_an", "lambda", "inner_gain", "inner_cost", "reported_lambda"]]
    fa = {}

    def at(sd, an, held, lam):
        if (sd, an) not in fa:
            fa[sd, an] = frontier_analysis(_mu_frontier(sd, an)).set_index(["held_out", "lambda"])
        return fa[sd, an].loc[(held, lam)]

    rows = []
    for x in sel.to_dict("records"):
        c, f = at(x["mu_sd"], x["mu_an"], x["outer"], x["lambda"]), at(0.5, 1.0, x["outer"], x["reported_lambda"])
        rows.append({**x, "heldout_gain": c.frag_gain, "heldout_cost": c.canonical_loss,
                     "fixed_heldout_gain": f.frag_gain, "fixed_heldout_cost": f.canonical_loss,
                     "gain_diff_vs_fixed": c.frag_gain - f.frag_gain,
                     "cost_diff_vs_fixed": c.canonical_loss - f.canonical_loss,
                     "is_fixed_choice": int(x["mu_sd"] == 0.5 and x["mu_an"] == 1.0),
                     "n_eval_rows": int(_mu_frontier(x["mu_sd"], x["mu_an"]).n_e1.max())})
    # mu_an alone (mu_sd = 0.5): mean held-out gain and cost over the five folds
    sens = []
    for an in (0.5, 1.0, 2.0):
        g = frontier_analysis(_mu_frontier(0.5, an))
        n = int(_mu_frontier(0.5, an).n_e1.max())
        for lam, h in g[g["lambda"] > 0].groupby("lambda"):
            sens.append({"mu_an": an, "lambda": lam, "folds": len(h), "gain": h.frag_gain.mean(),
                         "cost": h.canonical_loss.mean(), "n_eval_rows": n})
    return pd.DataFrame(rows), pd.DataFrame(sens)


def run() -> None:
    sel, sens = mu_nested()
    sel.to_csv(TABLES / "mu_nested_selection.csv", index=False)
    sens.to_csv(TABLES / "mu_nested_mu_an_sensitivity.csv", index=False)
    dose, canon, natrom = lolo_plain()
    dose.to_csv(TABLES / "lora_reseg_dose_response_lolo.csv", index=False)
    canon.to_csv(TABLES / "lora_canonical_regression_lolo.csv", index=False)
    natrom.to_csv(TABLES / "lora_native_romanised_lolo.csv", index=False)
    for name, f in [("", "frontier_plain.csv"), ("_anchored", "frontier_anchored.csv")]:
        fr = pd.read_csv(R / f)
        fr.to_csv(TABLES / f"lora_anchored_frontier{name}.csv", index=False)
        frontier_analysis(fr).to_csv(TABLES / f"lora_lambda_frontier_analysis{name}.csv", index=False)
    for lam in (0.5, 1.0):
        frontier_grid(lam).to_csv(TABLES / f"frontier_grid_lambda{lam}.csv", index=False)
    inner, agg, sel = nested_selection()
    inner.to_csv(TABLES / "nested_lambda_inner.csv", index=False)
    agg.to_csv(TABLES / "nested_lambda_validation.csv", index=False)
    sel.to_csv(TABLES / "nested_lambda_selection.csv", index=False)
    anchored_vs_plain_retention().to_csv(TABLES / "lora_anchored_vs_plain_retention.csv", index=False)
    zeroshot_repair().to_csv(TABLES / "transfer_zeroshot_repair.csv", index=False)
    transfer_lambda().to_csv(TABLES / "transfer_lambda_sweep.csv", index=False)
    transfer_lam075().to_csv(TABLES / "transfer_selected_lambda.csv", index=False)
    global ON_TARGET
    ON_TARGET = False
    try:
        zeroshot_repair().to_csv(TABLES / "transfer_zeroshot_repair_all_systems.csv", index=False)
        transfer_lam075().to_csv(TABLES / "transfer_selected_lambda_all_systems.csv", index=False)
    finally:
        ON_TARGET = True
    crossscript_fixed().to_csv(TABLES / "transfer_crossscript_training.csv", index=False)
    source_disjointness().to_csv(TABLES / "source_disjointness.csv", index=False)


if __name__ == "__main__":
    run()

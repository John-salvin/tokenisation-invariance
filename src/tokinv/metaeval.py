"""Section 5.1 (reliability of eleven metrics) and Section 5.3 (meta-evaluation).

Inputs: ``data/scores/metaeval_panel.parquet``, one row per (segment, script
condition) with eleven metric scores and the MQM score, and
``data/indic/segments.parquet`` for the chrF audit.

Design facts measured from the data, not assumed:
  * 6 MT systems per language, so C(6,2) = 15 system pairs.
  * Partially crossed: each (source, system) cell is unique; about 253 of 377
    sources per language carry >= 2 systems and about 188 carry all six.
  * MQM is identical across the native and romanised conditions (same
    translation, different script), so the human ranking is fixed and any
    ranking change is a pure metric artefact.

acc*_eq pairs are cross-system pairs of the same source segment, pooled with a
single global epsilon. Our sweep is checked against the vendored reference
implementation (vendor/tau_optimization.py) before the pooled number is used.
SPA needs a fully crossed matrix, so it uses the sources carrying all systems.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import stats
from .paths import DATA, INDIC, INDIC_LANGS, TABLES
from .vendor import pce as ref_pce
from .vendor import tau_optimization as ref_tau

PANEL = DATA / "scores" / "metaeval_panel.parquet"

NBOOT = 1000
NPERM = 1000
SEED = 20260919
CONDS = ["native", "romanised"]

# +1 higher-is-better, -1 lower-is-better
ORIENT = {"comet_22": +1, "bertscore": +1, "bleu": +1, "chrf": +1, "ter": -1,
          "bleurt20": +1, "metricx24": -1, "cometkiwi22": +1, "cometkiwi23": +1,
          "xcomet_xl": +1, "xcomet_xxl": +1}
METRICS = list(ORIENT)
NAMES = {"xcomet_xxl": "xCOMET-XXL", "xcomet_xl": "xCOMET-XL",
         "cometkiwi22": "CometKiwi-22", "metricx24": "MetricX-24",
         "cometkiwi23": "CometKiwi-23", "comet_22": "COMET-22",
         "bleurt20": "BLEURT-20", "bertscore": "BERTScore",
         "bleu": "BLEU", "ter": "TER", "chrf": "chrF"}
CHRF_MAX = 100.0


def load_panel() -> pd.DataFrame:
    p = pd.read_parquet(PANEL)
    return p[p.mqm_valid].copy()


# ------------------------------------------------------------ Table 3 ------ #
def reliability(panel: pd.DataFrame) -> pd.DataFrame:
    """Pooled |Spearman| of each metric with MQM, per script condition."""
    rows = []
    for cond in CONDS:
        for col, name in NAMES.items():
            s = panel[panel.condition == cond][[col, "mqm_score"]].dropna()
            if col == "chrf" and cond == "native":
                # 37 Tamil native chrF values exceed the metric's own maximum
                # of 100 (Appendix "Data Quality"); they are excluded here and
                # the all-rows value is reported in chrf_native_rho.csv.
                s = s[s.chrf <= CHRF_MAX]
            rows.append({"condition": cond, "metric": name,
                         "recomputed": round(abs(stats.spearman(s[col], s.mqm_score)), 5),
                         "n": len(s)})
    return pd.DataFrame(rows)


def chrf_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    seg = pd.read_parquet(INDIC / "segments.parquet")
    seg["lang"] = seg["lang"].astype(str)
    seg["condition"] = seg["condition"].astype(str)
    rows = []
    for (lang, cond), g in seg.groupby(["lang", "condition"]):
        bad = g[g.chrf > CHRF_MAX]
        rows.append({"lang": lang, "condition": cond, "n_segments": len(g),
                     "n_chrf_over_100": len(bad),
                     "pct_chrf_over_100": 100.0 * len(bad) / len(g),
                     "max_chrf": float(g.chrf.max()), "min_chrf": float(g.chrf.min())})
    nat = seg[(seg.condition == "native") & seg.mqm_valid]
    rho = []
    for name, g in {"all_rows": nat, "chrf_le_100": nat[nat.chrf <= CHRF_MAX]}.items():
        r = stats.spearman(g.chrf, g.mqm_score)
        rho.append({"population": name, "n": len(g),
                    "spearman_chrf_mqm": r, "abs_spearman": abs(r)})
    return (pd.DataFrame(rows).sort_values(["lang", "condition"]).reset_index(drop=True),
            pd.DataFrame(rho))


# ------------------------------------------------------------- acc*_eq ----- #
def _pair_arrays(dh, dm):
    """Below the threshold a pair is correct iff concordant; at or above it
    (metric tie) iff the human also ties. Mirrors the reference _RankedPair."""
    d = np.abs(dm)
    correct_below = ((dh > 0) & (dm > 0)) | ((dh < 0) & (dm < 0))
    correct_above = dh == 0
    return d, correct_below.astype(np.int64), correct_above.astype(np.int64)


def acc_star_eq_pooled(dh, dm):
    """Pooled acc*_eq with a single global epsilon. Returns (acc, eps)."""
    n = len(dh)
    if n == 0:
        return float("nan"), float("nan")
    d, cb, ca = _pair_arrays(dh, dm)
    o = np.argsort(d, kind="mergesort")
    d, cb, ca = d[o], cb[o], ca[o]
    cb_cum = np.concatenate([[0], np.cumsum(cb)])
    ca_cum = np.concatenate([[0], np.cumsum(ca)])
    ks = np.searchsorted(d, np.unique(d), side="right")
    ks = np.unique(np.concatenate([[np.searchsorted(d, 0.0, side="right")], ks]))
    accs = (ca_cum[ks] + (cb_cum[-1] - cb_cum[ks])) / n
    i = int(np.nanargmax(accs))
    k = ks[i]
    return float(accs[i]), (0.0 if k == 0 else float(d[k - 1]))


def acc_star_eq_rowavg(rows_dh, rows_dm):
    """Per-source-averaged acc*_eq: the quantity the reference returns. Used
    only to validate our pair classification against it."""
    parts = [(_pair_arrays(dh, dm), r) for r, (dh, dm) in enumerate(zip(rows_dh, rows_dm))]
    d = np.concatenate([p[0][0] for p in parts])
    cb = np.concatenate([p[0][1] for p in parts])
    ca = np.concatenate([p[0][2] for p in parts])
    row = np.concatenate([np.full(len(p[0][0]), p[1]) for p in parts])
    npair_row = np.array([len(x) for x in rows_dh], dtype=np.float64)
    o = np.argsort(d, kind="mergesort")
    d, cb, ca, row = d[o], cb[o], ca[o], row[o]
    cur = np.zeros(len(rows_dh))
    np.add.at(cur, row, cb)
    best, best_eps = (cur / npair_row).mean(), 0.0
    i, n = 0, len(d)
    while i < n:
        j = i
        while j < n and d[j] == d[i]:
            j += 1
        np.add.at(cur, row[i:j], ca[i:j] - cb[i:j])
        val = (cur / npair_row).mean()
        if d[i] == 0.0:
            best, best_eps = val, 0.0
        elif val > best:
            best, best_eps = val, float(d[i])
        i = j
    return float(best), float(best_eps)


# ----------------------------------------------------------------- SPA ----- #
def _pvals(seg_scores, perm):
    seg = seg_scores.astype(np.float32)
    sys_scores = seg.sum(axis=1)
    partial = perm @ seg.T
    nsys = seg.shape[0]
    p = np.full((nsys, nsys), np.nan)
    for a in range(nsys):
        for b in range(a + 1, nsys):
            p[a, b] = np.mean(partial[:, a] - partial[:, b] >= sys_scores[a] - sys_scores[b])
    return p


def spa(human_mat, metric_mat, perm):
    """1 - PCE with the reference estimator."""
    return float(ref_pce.compute_one_minus_pce(_pvals(human_mat, perm), _pvals(metric_mat, perm)))


def make_perm(nperm, nseg, rng):
    m = rng.random(size=(nperm, nseg), dtype=np.float32)
    np.rint(m, out=m, casting="same_kind")
    m *= 2.0
    m -= 1.0
    return m


def qn_map(scores, reference_sorted):
    """Quantile normalisation: rank -> quantile of a reference distribution."""
    r = pd.Series(scores).rank().to_numpy()
    return np.quantile(reference_sorted, r / (len(scores) + 1))


def metaeval(panel: pd.DataFrame):
    """acc*_eq, SPA and ranking flips. One generator is shared across all
    bootstraps in a fixed loop order, so the order below must not change."""
    rng = np.random.default_rng(SEED)
    out_acc, out_spa, out_flip, validation = [], [], [], []

    for lang in INDIC_LANGS:
        for cond in CONDS:
            cell = panel[(panel.lang == lang) & (panel.condition == cond)]
            groups = [g for _, g in cell.groupby("source_en") if len(g) >= 2]
            for metric in METRICS:
                sgn = ORIENT[metric]
                rows_dh, rows_dm, rows_src = [], [], []
                for gi, g in enumerate(groups):
                    m = g[metric].to_numpy(dtype=float)
                    h = g.mqm_score.to_numpy(dtype=float)
                    ok = ~np.isnan(m) & ~np.isnan(h)
                    m, h = m[ok] * sgn, h[ok]
                    if len(m) < 2:
                        continue
                    ii, jj = np.triu_indices(len(m), 1)
                    rows_dh.append(h[ii] - h[jj])
                    rows_dm.append(m[ii] - m[jj])
                    rows_src.append(gi)
                if not rows_dh:
                    continue
                dh, dm = np.concatenate(rows_dh), np.concatenate(rows_dm)
                src = np.concatenate([np.full(len(a), s) for a, s in zip(rows_dh, rows_src)])
                acc, eps = acc_star_eq_pooled(dh, dm)
                uniq = np.unique(src)
                idx = {s: np.where(src == s)[0] for s in uniq}
                boots = np.empty(NBOOT)
                for b in range(NBOOT):
                    sel = np.concatenate([idx[s] for s in rng.choice(uniq, size=len(uniq), replace=True)])
                    boots[b], _ = acc_star_eq_pooled(dh[sel], dm[sel])
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                out_acc.append({"metric": metric, "lang": lang, "condition": cond,
                                "acc_star_eq": acc, "epsilon": eps, "ci_lo": lo, "ci_hi": hi,
                                "n_pairs": int(len(dh)), "n_sources": int(len(uniq)),
                                "n_boot": NBOOT})
                if lang == "guj" and cond == "native" and metric in ("comet_22", "metricx24"):
                    mine, mine_eps = acc_star_eq_rowavg(rows_dh, rows_dm)
                    width = max(len(g) for g in groups)
                    H = np.full((len(groups), width), None, dtype=object)
                    M = np.full((len(groups), width), None, dtype=object)
                    for gi, g in enumerate(groups):
                        mv = g[metric].to_numpy(dtype=float) * sgn
                        hv = g.mqm_score.to_numpy(dtype=float)
                        for k in range(len(g)):
                            if not (np.isnan(mv[k]) or np.isnan(hv[k])):
                                H[gi, k], M[gi, k] = float(hv[k]), float(mv[k])
                    res = ref_tau.tau_optimization(M, H, ref_tau.TauSufficientStats.acc_23,
                                                   sample_rate=1.0)
                    validation.append({"metric": metric, "lang": lang, "condition": cond,
                                       "mine_rowavg": mine, "reference_rowavg": float(res.best_tau),
                                       "abs_diff": abs(mine - float(res.best_tau)),
                                       "mine_eps": mine_eps,
                                       "reference_eps": float(res.best_threshold),
                                       "pooled_for_contrast": acc})

    for lang in INDIC_LANGS:
        for cond in CONDS:
            cell = panel[(panel.lang == lang) & (panel.condition == cond)]
            systems = sorted(cell.system.unique())
            for metric in METRICS:
                sub = cell[["source_en", "system", "mqm_score", metric]].dropna()
                w = sub.pivot_table(index="source_en", columns="system",
                                    values=[metric, "mqm_score"], aggfunc="first")
                full = w.dropna()
                if len(full) < 10:
                    continue
                sysnames = [s for s in systems if (metric, s) in w.columns]
                Hm = full["mqm_score"][sysnames].to_numpy().T
                Mm = full[metric][sysnames].to_numpy().T * ORIENT[metric]
                perm = make_perm(NPERM, Hm.shape[1], np.random.default_rng(SEED))
                val = spa(Hm, Mm, perm)
                bs = np.empty(NBOOT)
                for b in range(NBOOT):
                    pick = rng.integers(0, Hm.shape[1], Hm.shape[1])
                    bs[b] = spa(Hm[:, pick], Mm[:, pick], perm)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                out_spa.append({"metric": metric, "lang": lang, "condition": cond,
                                "spa": val, "ci_lo": lo, "ci_hi": hi,
                                "n_systems": len(sysnames), "n_sources_full": int(len(full)),
                                "n_boot": NBOOT, "n_perm": NPERM})

    for metric in METRICS:
        ref_sorted = np.sort(panel[panel.condition == "native"][metric].dropna().to_numpy())
        for lang in INDIC_LANGS:
            means = {}
            for cond in CONDS:
                cell = panel[(panel.lang == lang) & (panel.condition == cond)][["system", metric]].dropna()
                if cell.empty:
                    continue
                raw = cell[metric].to_numpy(dtype=float)
                d = pd.DataFrame({"system": cell.system.to_numpy(),
                                  "raw": raw * ORIENT[metric],
                                  "qn": qn_map(raw, ref_sorted) * ORIENT[metric]})
                means[cond] = d.groupby("system").mean()
            if len(means) != 2:
                continue
            common = sorted(set(means["native"].index) & set(means["romanised"].index))
            for scheme in ("raw", "qn"):
                flipped = []
                pairs = list(itertools.combinations(common, 2))
                for a, b in pairs:
                    na = means["native"].loc[a, scheme] - means["native"].loc[b, scheme]
                    ra = means["romanised"].loc[a, scheme] - means["romanised"].loc[b, scheme]
                    if np.sign(na) != np.sign(ra):
                        flipped.append(f"{a}>{b}" if na > 0 else f"{b}>{a}")
                out_flip.append({"metric": metric, "lang": lang, "scheme": scheme,
                                 "n_systems": len(common), "n_pairs": len(pairs),
                                 "n_flips": len(flipped), "flipped_pairs": ";".join(flipped)})

    return (pd.DataFrame(out_acc), pd.DataFrame(out_spa), pd.DataFrame(out_flip),
            pd.DataFrame(validation))


def acc_floor(panel: pd.DataFrame) -> pd.DataFrame:
    """The acc*_eq floor: predicting "tie" for every pair scores P(human tie),
    so a metric at the floor carries no ranking information."""
    rows = []
    for lang in INDIC_LANGS:
        for cond in CONDS:
            cell = panel[(panel.lang == lang) & (panel.condition == cond)]
            dh = []
            for _, g in cell.groupby("source_en"):
                h = g.mqm_score.to_numpy(float)
                h = h[~np.isnan(h)]
                if len(h) < 2:
                    continue
                i, j = np.triu_indices(len(h), 1)
                dh.append(h[i] - h[j])
            dh = np.concatenate(dh)
            rows.append({"lang": lang, "condition": cond, "n_pairs": len(dh),
                         "p_human_tie": float((dh == 0).mean()),
                         "acc_floor_all_tie": float((dh == 0).mean())})
    return pd.DataFrame(rows)


def run() -> None:
    panel = load_panel()
    reliability(panel).to_csv(TABLES / "verify_reliability_table.csv", index=False)
    oor, rho = chrf_audit()
    oor.to_csv(TABLES / "chrf_out_of_range.csv", index=False)
    rho.to_csv(TABLES / "chrf_native_rho.csv", index=False)
    acc_floor(panel).to_csv(TABLES / "metaeval_acc_floor.csv", index=False)
    acc, spa_, flips, val = metaeval(panel)
    if not (val.abs_diff < 1e-12).all():
        raise AssertionError(f"acc*_eq disagrees with the reference:\n{val}")
    acc.to_csv(TABLES / "metaeval_acc_star_eq.csv", index=False)
    spa_.to_csv(TABLES / "metaeval_spa.csv", index=False)
    flips.to_csv(TABLES / "system_ranking_flips.csv", index=False)
    val.to_csv(TABLES / "metaeval_acc_validation.csv", index=False)


if __name__ == "__main__":
    run()


# ------------------------------------- App. Meta-Evaluation Table ---- #
CAPACITY_B = {"COMET-22": 0.56, "xCOMET-XL": 3.5, "xCOMET-XXL": 10.7}


def _rho_cell(panel, col, lang, cond):
    s = panel[(panel.condition == cond)]
    if lang is not None:
        s = s[s.lang == lang]
    s = s[[col, "mqm_score"]].dropna()
    if col == "chrf" and cond == "native":
        s = s[s.chrf <= CHRF_MAX]
    return ORIENT[col] * stats.spearman(s[col], s.mqm_score), len(s)


def reliability_ranking(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-language and pooled agreement, orientation-corrected (higher is
    better for every metric), so a genuinely inverted cell stays negative."""
    rows = []
    for col, name in NAMES.items():
        r = {"metric": name, "lower_is_better": ORIENT[col] < 0}
        for cond in CONDS:
            for lang in INDIC_LANGS:
                r[f"rho_{cond}_{lang}"], r[f"n_{cond}_{lang}"] = _rho_cell(panel, col, lang, cond)
        for cond in CONDS:
            r[f"pooled_rho_{cond}"], r[f"pooled_n_{cond}"] = _rho_cell(panel, col, None, cond)
        rows.append(r)
    d = pd.DataFrame(rows)
    d["native_rank"] = d.pooled_rho_native.rank(ascending=False).astype(int)
    d["romanised_rank"] = d.pooled_rho_romanised.rank(ascending=False).astype(int)
    d["degradation_pooled"] = d.pooled_rho_native - d.pooled_rho_romanised
    d["rho_retained_pct"] = 100 * d.pooled_rho_romanised / d.pooled_rho_native
    return d.sort_values("native_rank").reset_index(drop=True)


def capacity_pattern(ranking: pd.DataFrame) -> pd.DataFrame:
    """Degradation within the COMET family as capacity grows: the mean over
    languages of the per-language drop, not the drop of pooled values."""
    rows = []
    for name, b in CAPACITY_B.items():
        r = ranking.set_index("metric").loc[name]
        nat = np.array([r[f"rho_native_{l}"] for l in INDIC_LANGS])
        rom = np.array([r[f"rho_romanised_{l}"] for l in INDIC_LANGS])
        rows.append({"metric": name, "mean_degradation_abs_rho": float(np.mean(nat - rom)),
                     "mean_relative_degradation": float(np.mean((nat - rom) / nat)),
                     "capacity_b_params": b})
    d = pd.DataFrame(rows)
    d["monotonic_abs_pooled"] = d.mean_degradation_abs_rho.is_monotonic_increasing
    d["monotonic_rel_pooled"] = d.mean_relative_degradation.is_monotonic_increasing
    return d


def run_capacity() -> None:
    rk = reliability_ranking(load_panel())
    rk.to_csv(TABLES / "metric_reliability_ranking.csv", index=False)
    capacity_pattern(rk).to_csv(TABLES / "capacity_pattern_pooled.csv", index=False)

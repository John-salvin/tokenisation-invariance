"""Meta-evaluation of the IndicMT Eval metric panel: acc*_eq, SPA and ranking flips.

Ran on the GPU cluster; see experiments/README.md.
"""
import sys, os, json, itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from vendor import tau_optimization as ref_tau
from vendor import pce as ref_pce

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
T = f"{REPO}/results/tables"
NBOOT = int(os.environ.get("NBOOT", "1000"))
NPERM = int(os.environ.get("NPERM", "1000"))
SEED = 20260919

LANGS = ["guj", "tam", "mal", "mar", "hin"]
CONDS = ["native", "romanised"]
ORIENT = {"comet_22": +1, "bertscore": +1, "bleu": +1, "chrf": +1, "ter": -1,
          "bleurt20": +1, "metricx24": -1, "cometkiwi22": +1, "cometkiwi23": +1,
          "xcomet_xl": +1, "xcomet_xxl": +1}
METRICS = list(ORIENT.keys())


def _pair_arrays(dh, dm):
    d = np.abs(dm)
    correct_below = ((dh > 0) & (dm > 0)) | ((dh < 0) & (dm < 0))
    correct_above = (dh == 0)
    return d, correct_below.astype(np.int64), correct_above.astype(np.int64)


def acc_star_eq_pooled(dh, dm):
    n = len(dh)
    if n == 0:
        return float("nan"), float("nan")
    d, cb, ca = _pair_arrays(dh, dm)
    o = np.argsort(d, kind="mergesort")
    d, cb, ca = d[o], cb[o], ca[o]
    cb_cum = np.concatenate([[0], np.cumsum(cb)])
    ca_cum = np.concatenate([[0], np.cumsum(ca)])
    total_cb = cb_cum[-1]
    ks = np.searchsorted(d, np.unique(d), side="right")
    ks = np.unique(np.concatenate([[np.searchsorted(d, 0.0, side="right")], ks]))
    accs = (ca_cum[ks] + (total_cb - cb_cum[ks])) / n
    i = int(np.nanargmax(accs))
    k = ks[i]
    eps = 0.0 if k == 0 else float(d[k - 1])
    return float(accs[i]), eps


def acc_star_eq_rowavg(rows_dh, rows_dm):
    all_d, all_cb, all_ca, all_row = [], [], [], []
    for r, (dh, dm) in enumerate(zip(rows_dh, rows_dm)):
        d, cb, ca = _pair_arrays(dh, dm)
        all_d.append(d); all_cb.append(cb); all_ca.append(ca)
        all_row.append(np.full(len(d), r))
    d = np.concatenate(all_d); cb = np.concatenate(all_cb)
    ca = np.concatenate(all_ca); row = np.concatenate(all_row)
    nrows = len(rows_dh)
    npair_row = np.array([len(x) for x in rows_dh], dtype=np.float64)

    o = np.argsort(d, kind="mergesort")
    d, cb, ca, row = d[o], cb[o], ca[o], row[o]
    cur = np.zeros(nrows)
    np.add.at(cur, row, cb)
    best = (cur / npair_row).mean()
    best_eps = 0.0
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


def spa(human_mat, metric_mat, perm):
    ph = _pvals(human_mat, perm)
    pm = _pvals(metric_mat, perm)
    return float(ref_pce.compute_one_minus_pce(ph, pm))


def _pvals(seg_scores, perm):
    seg = seg_scores.astype(np.float32)
    sys_scores = seg.sum(axis=1)
    partial = perm @ seg.T
    nsys = seg.shape[0]
    p = np.full((nsys, nsys), np.nan)
    for a in range(nsys):
        for b in range(a + 1, nsys):
            null_d = partial[:, a] - partial[:, b]
            p[a, b] = np.mean(null_d >= (sys_scores[a] - sys_scores[b]))
    return p


def make_perm(nperm, nseg, rng):
    m = rng.random(size=(nperm, nseg), dtype=np.float32)
    np.rint(m, out=m, casting="same_kind")
    m *= 2.0
    m -= 1.0
    return m


def qn_map(scores, reference_sorted):
    r = pd.Series(scores).rank().to_numpy()
    frac = r / (len(scores) + 1)
    return np.quantile(reference_sorted, frac)


def main():
    rng = np.random.default_rng(SEED)
    panel = pd.read_csv(f"{T}/metaeval_panel.csv")
    panel = panel[panel["mqm_valid"]].copy()

    out_acc, out_spa, out_flip, validation = [], [], [], []

    for lang in LANGS:
        for cond in CONDS:
            cell = panel[(panel["lang"] == lang) & (panel["condition"] == cond)]
            groups = [g for _, g in cell.groupby("source_en") if len(g) >= 2]
            src_ids = np.arange(len(groups))

            for metric in METRICS:
                sgn = ORIENT[metric]
                rows_dh, rows_dm, rows_src = [], [], []
                for gi, g in enumerate(groups):
                    m = g[metric].to_numpy(dtype=float)
                    h = g["mqm_score"].to_numpy(dtype=float)
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
                dh = np.concatenate(rows_dh); dm = np.concatenate(rows_dm)
                src = np.concatenate([np.full(len(a), s) for a, s in zip(rows_dh, rows_src)])
                acc, eps = acc_star_eq_pooled(dh, dm)

                uniq = np.unique(src)
                idx_by_src = {s: np.where(src == s)[0] for s in uniq}
                boots = np.empty(NBOOT)
                for b in range(NBOOT):
                    pick = rng.choice(uniq, size=len(uniq), replace=True)
                    sel = np.concatenate([idx_by_src[s] for s in pick])
                    boots[b], _ = acc_star_eq_pooled(dh[sel], dm[sel])
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                out_acc.append({"metric": metric, "lang": lang, "condition": cond,
                                "acc_star_eq": acc, "epsilon": eps,
                                "ci_lo": lo, "ci_hi": hi,
                                "n_pairs": int(len(dh)), "n_sources": int(len(uniq)),
                                "n_boot": NBOOT})

                if lang == "guj" and cond == "native" and metric in ("comet_22", "metricx24"):
                    mine, mine_eps = acc_star_eq_rowavg(rows_dh, rows_dm)
                    width = max(len(g) for g in groups)
                    H = np.full((len(groups), width), None, dtype=object)
                    M = np.full((len(groups), width), None, dtype=object)
                    for gi, g in enumerate(groups):
                        mv = g[metric].to_numpy(dtype=float) * sgn
                        hv = g["mqm_score"].to_numpy(dtype=float)
                        for k in range(len(g)):
                            if not (np.isnan(mv[k]) or np.isnan(hv[k])):
                                H[gi, k] = float(hv[k]); M[gi, k] = float(mv[k])
                    res = ref_tau.tau_optimization(
                        M, H, ref_tau.TauSufficientStats.acc_23, sample_rate=1.0)
                    validation.append({
                        "metric": metric, "lang": lang, "condition": cond,
                        "mine_rowavg": mine, "reference_rowavg": float(res.best_tau),
                        "abs_diff": abs(mine - float(res.best_tau)),
                        "mine_eps": mine_eps, "reference_eps": float(res.best_threshold),
                        "pooled_for_contrast": acc})

    pd.DataFrame(out_acc).to_csv(f"{T}/metaeval_acc_star_eq.csv", index=False)
    pd.DataFrame(validation).to_csv(f"{T}/metaeval_acc_validation.csv", index=False)
    print("== validation vs vendored reference ==")
    print(pd.DataFrame(validation).to_string(index=False))

    for lang in LANGS:
        for cond in CONDS:
            cell = panel[(panel["lang"] == lang) & (panel["condition"] == cond)]
            systems = sorted(cell["system"].unique())
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
    pd.DataFrame(out_spa).to_csv(f"{T}/metaeval_spa.csv", index=False)

    for metric in METRICS:
        natall = panel[(panel["condition"] == "native")][metric].dropna().to_numpy()
        ref_sorted = np.sort(natall)
        for lang in LANGS:
            means = {}
            for cond in CONDS:
                cell = panel[(panel["lang"] == lang) & (panel["condition"] == cond)]
                cell = cell[["system", metric]].dropna()
                if cell.empty:
                    continue
                raw = cell[metric].to_numpy(dtype=float)
                qn = qn_map(raw, ref_sorted)
                d = pd.DataFrame({"system": cell["system"].to_numpy(),
                                  "raw": raw * ORIENT[metric],
                                  "qn": qn * ORIENT[metric]})
                means[cond] = d.groupby("system").mean()
            if len(means) != 2:
                continue
            common = sorted(set(means["native"].index) & set(means["romanised"].index))
            for scheme in ("raw", "qn"):
                flips, tot = 0, 0
                flipped_pairs = []
                for a, b in itertools.combinations(common, 2):
                    na = means["native"].loc[a, scheme] - means["native"].loc[b, scheme]
                    ra = means["romanised"].loc[a, scheme] - means["romanised"].loc[b, scheme]
                    tot += 1
                    if np.sign(na) != np.sign(ra):
                        flips += 1
                        flipped_pairs.append(f"{a}>{b}" if na > 0 else f"{b}>{a}")
                out_flip.append({"metric": metric, "lang": lang, "scheme": scheme,
                                 "n_systems": len(common), "n_pairs": tot,
                                 "n_flips": flips,
                                 "flipped_pairs": ";".join(flipped_pairs)})
    pd.DataFrame(out_flip).to_csv(f"{T}/system_ranking_flips.csv", index=False)

    print("\n== wrote metaeval_acc_star_eq.csv / metaeval_spa.csv / system_ranking_flips.csv ==")
    print(f"acc rows={len(out_acc)} spa rows={len(out_spa)} flip rows={len(out_flip)}")


if __name__ == "__main__":
    main()

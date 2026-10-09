"""Section 7 and App. Quantile Normalisation Across the Panel.

QN maps each score to the quantile of a reference distribution (pooled native
scores) at its own rank. It is monotone within every (language, condition)
cell, so it cannot change within-cell ranking; it can only remove offsets
between cells. That is the paper's test for which fault is repairable post hoc:
the cross-script offset is, the within-script sensitivity is not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats
from .metaeval import PANEL
from .paths import INDIC, INDIC_LANGS, LADDER8, TABLES, DATA

CONDS = ["native", "romanised"]

# (metric label, column, source, lower_is_better, expected n per language).
# Surface metrics and COMET-22 come from segments.parquet; learned metrics from
# the scored panel. xCOMET was scored on a 300-segment subset per language.
PANEL_METRICS = [
    ("COMET-22", "comet_22", "seg", False, 1400),
    ("BERTScore", "bertscore", "seg", False, 1400),
    ("BLEURT (legacy, segments.parquet)", "bleurt", "seg", False, 1400),
    ("chrF (surface floor)", "chrf", "seg", False, 1400),
    ("BLEU (surface floor)", "bleu", "seg", False, 1400),
    ("TER (surface floor)", "ter", "seg", True, 1400),
    ("CometKiwi-22", "cometkiwi22", "panel", False, 1400),
    ("CometKiwi-23", "cometkiwi23", "panel", False, 1400),
    ("xCOMET-XL", "xcomet_xl", "panel", False, 300),
    ("xCOMET-XXL", "xcomet_xxl", "panel", False, 300),
    ("MetricX-24", "metricx24", "panel", True, 1400),
    ("BLEURT-20", "bleurt20", "panel", False, 1400),
]
LOLO_ORDER = ["COMET-22", "CometKiwi-22", "CometKiwi-23", "xCOMET-XL", "xCOMET-XXL",
              "BERTScore", "MetricX-24", "BLEURT-20", "chrF (surface floor)",
              "BLEU (surface floor)", "TER (surface floor)"]


def qn_map(scores, reference_sorted):
    r = pd.Series(np.asarray(scores)).rank(method="average").to_numpy()
    return np.quantile(reference_sorted, r / (len(scores) + 1))


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def _frames():
    seg = pd.read_parquet(INDIC / "segments.parquet")
    seg = seg[seg.mqm_valid].copy()
    seg["lang"] = seg["lang"].astype(str)
    seg["condition"] = seg["condition"].astype(str)
    panel = pd.read_parquet(PANEL)
    return seg, panel


def _metric_frame(seg, panel, col, src):
    d = (seg if src == "seg" else panel)[["lang", "condition", col, "mqm_score"]]
    return d.rename(columns={col: "score", "mqm_score": "H"}).dropna()


def in_sample_panel() -> pd.DataFrame:
    seg, panel = _frames()
    rows = []
    for name, col, src, lib, n_exp in PANEL_METRICS:
        df = _metric_frame(seg, panel, col, src)
        ref = np.sort(df[df.condition == "native"].score.to_numpy())
        parts = []
        for lang in INDIC_LANGS:
            for cond in CONDS:
                s = df[(df.lang == lang) & (df.condition == cond)]
                parts.append(pd.DataFrame({"lang": lang, "cond": cond, "raw": s.score.to_numpy(),
                                           "qn": qn_map(s.score, ref), "H": s.H.to_numpy()}))
        Q = pd.concat(parts, ignore_index=True)
        inv = max(abs(stats.spearman(c.raw, c.H) - stats.spearman(c.qn, c.H))
                  for _, c in Q.groupby(["lang", "cond"]))
        gap = lambda v: np.mean([abs(Q[(Q.lang == l) & (Q.cond == "native")][v].mean()
                                     - Q[(Q.lang == l) & (Q.cond == "romanised")][v].mean())
                                 for l in INDIC_LANGS])
        sb, sa = stats.spearman(Q.raw, Q.H), stats.spearman(Q.qn, Q.H)
        pb, pa = pearson(Q.raw, Q.H), pearson(Q.qn, Q.H)
        rows.append({"metric": name, "lower_is_better": lib, "n_langs": 5, "n_total": len(Q),
                     "n_per_lang_native_mean": Q[Q.cond == "native"].groupby("lang").size().mean(),
                     "n_expected_per_lang_full_corpus": n_exp,
                     "max_within_cell_invariance_diff": inv,
                     "pooled_spearman_before": sb, "pooled_spearman_after": sa,
                     "spearman_gain": sa - sb, "pooled_pearson_before": pb,
                     "pooled_pearson_after": pa, "pearson_gain": pa - pb,
                     "gap_before": gap("raw"), "gap_after": gap("qn")})
    return pd.DataFrame(rows)


def lolo_panel(in_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate the reference on four languages' native scores, apply to the
    fifth language's native and romanised scores. |rho| for error metrics."""
    seg, panel = _frames()
    ins = in_sample.copy()
    ins["g"] = np.where(ins.lower_is_better,
                        ins.pooled_spearman_after.abs() - ins.pooled_spearman_before.abs(),
                        ins.spearman_gain)
    in_gain = dict(zip(ins.metric, ins.g))
    rows = []
    for name, col, src, lib, n_exp in PANEL_METRICS:
        if name.startswith("BLEURT (legacy"):
            continue
        df = _metric_frame(seg, panel, col, src)
        cn, cr = df[df.condition == "native"], df[df.condition == "romanised"]
        for held in INDIC_LANGS:
            calib = [l for l in INDIC_LANGS if l != held]
            ref = np.sort(np.concatenate([cn[cn.lang == l].score.to_numpy() for l in calib]))
            hn, hr = cn[cn.lang == held], cr[cr.lang == held]
            qn_n, qn_r = qn_map(hn.score, ref), qn_map(hr.score, ref)
            raw_all = np.concatenate([hn.score, hr.score])
            H_all = np.concatenate([hn.H, hr.H])
            b, a = stats.spearman(raw_all, H_all), stats.spearman(np.concatenate([qn_n, qn_r]), H_all)
            if lib:
                b, a = abs(b), abs(a)
            rows.append({"metric": name, "held_out_lang": held, "calib_langs": ",".join(calib),
                         "n_calib_reference": len(ref), "n_held_out": len(raw_all),
                         "n_expected_full_corpus": n_exp, "lolo_spearman_before": b,
                         "lolo_spearman_after": a, "lolo_gain": a - b,
                         "within_native_invariance_diff":
                             abs(stats.spearman(hn.score, hn.H) - stats.spearman(qn_n, hn.H)),
                         "in_sample_gain": in_gain[name]})
    detail = pd.DataFrame(rows)
    summary = detail.groupby("metric", sort=False).agg(
        n_folds=("held_out_lang", "count"), mean_n_held_out=("n_held_out", "mean"),
        lolo_gain_mean=("lolo_gain", "mean"), lolo_gain_min=("lolo_gain", "min"),
        lolo_gain_max=("lolo_gain", "max"),
        n_folds_negative=("lolo_gain", lambda s: int((s < 0).sum())),
        max_within_native_invariance_diff=("within_native_invariance_diff", "max"),
    ).reset_index()
    summary["in_sample_gain"] = summary.metric.map(in_gain)
    summary["n_expected_full_corpus"] = summary.metric.map(
        dict(zip(detail.metric, detail.n_expected_full_corpus)))
    summary["metric"] = pd.Categorical(summary.metric, LOLO_ORDER, ordered=True)
    return summary.sort_values("metric"), detail


def decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    """How much of COMET-22's pooled gain comes from removing the condition
    offset versus the language offset (Section 7)."""
    seg, _ = _frames()
    ref = np.sort(seg[seg.condition == "native"].comet_22.to_numpy())
    cell = {(l, c): seg[(seg.lang == l) & (seg.condition == c)]
            for l in INDIC_LANGS for c in CONDS}
    full, inv = [], []
    for (l, c), s in cell.items():
        q = qn_map(s.comet_22, ref)
        full.append(pd.DataFrame({"raw": s.comet_22.to_numpy(), "qn": q, "H": s.mqm_score.to_numpy()}))
        rb, ra = stats.spearman(s.comet_22, s.mqm_score), stats.spearman(q, s.mqm_score)
        inv.append({"lang": l, "cond": c, "n": len(s), "rho_before": rb,
                    "rho_after": ra, "abs_diff": abs(rb - ra)})
    Q = pd.concat(full, ignore_index=True)
    # condition offset only: each language onto its own native distribution
    A = pd.concat([pd.DataFrame({"v": qn_map(cell[(l, c)].comet_22,
                                             np.sort(cell[(l, "native")].comet_22.to_numpy())),
                                 "H": cell[(l, c)].mqm_score.to_numpy()})
                   for l in INDIC_LANGS for c in CONDS])
    # language offset only: one within-language rank space onto the global reference
    B = pd.concat([pd.DataFrame({
        "v": qn_map(np.concatenate([cell[(l, "native")].comet_22, cell[(l, "romanised")].comet_22]), ref),
        "H": np.concatenate([cell[(l, "native")].mqm_score, cell[(l, "romanised")].mqm_score])})
        for l in INDIC_LANGS])
    d = pd.DataFrame([
        {"variant": "before (raw)", "spearman": stats.spearman(Q.raw, Q.H), "pearson": pearson(Q.raw, Q.H)},
        {"variant": "condition-offset-only QN (per-language native ref)",
         "spearman": stats.spearman(A.v, A.H), "pearson": pearson(A.v, A.H)},
        {"variant": "language-offset-only QN (global pooled-native ref, within-lang ranks preserved)",
         "spearman": stats.spearman(B.v, B.H), "pearson": pearson(B.v, B.H)},
        {"variant": "full QN (both offsets removed, published)",
         "spearman": stats.spearman(Q.qn, Q.H), "pearson": pearson(Q.qn, Q.H)},
    ])
    d["spearman_gain_over_before"] = d.spearman - d.spearman.iloc[0]
    d["pearson_gain_over_before"] = d.pearson - d.pearson.iloc[0]
    return d, pd.DataFrame(inv)


def reseg_buckets() -> pd.DataFrame:
    """QN across fragmentation buckets, the boundary test: the least
    fragmented bucket plays the role of "native"."""
    e = pd.read_parquet(DATA / "scores" / "resegmentation" / "indic_comet22.parquet")
    e = e[(e.direction == "forward") & e.bucket.isin(LADDER8)]
    ref = np.sort(e[e.bucket == LADDER8[0]].comet22.to_numpy())
    parts, rows = [], []
    for b in LADDER8:
        s = e[e.bucket == b]
        q = qn_map(s.comet22, ref)
        parts.append(pd.DataFrame({"raw": s.comet22.to_numpy(), "qn": q, "H": s.mqm_score.to_numpy()}))
        rb, ra = stats.spearman(s.comet22, s.mqm_score), stats.spearman(q, s.mqm_score)
        rows.append({"bucket": b, "n": len(s), "rho_before": rb, "rho_after": ra,
                     "abs_diff": abs(rb - ra), "raw_mean": s.comet22.mean(), "qn_mean": q.mean()})
    Q = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame(rows)
    out["pooled_spearman_before"] = stats.spearman(Q.raw, Q.H)
    out["pooled_spearman_after"] = stats.spearman(Q.qn, Q.H)
    out["pooled_pearson_before"] = pearson(Q.raw, Q.H)
    out["pooled_pearson_after"] = pearson(Q.qn, Q.H)
    return out


def run() -> None:
    ins = in_sample_panel()
    ins.to_csv(TABLES / "comet_qn_panel.csv", index=False)
    s, d = lolo_panel(ins)
    s.to_csv(TABLES / "comet_qn_lolo_panel.csv", index=False)
    d.to_csv(TABLES / "comet_qn_lolo_panel_detail.csv", index=False)
    dec, inv = decomposition()
    dec.to_csv(TABLES / "comet_qn_decomposition.csv", index=False)
    inv.to_csv(TABLES / "comet_qn_invariance.csv", index=False)
    reseg_buckets().to_csv(TABLES / "comet_qn_reseg_buckets.csv", index=False)
    reseg_buckets_all_settings().to_csv(TABLES / "comet_qn_reseg_buckets_all_settings.csv", index=False)
    reseg_buckets_all_settings(False).to_csv(TABLES / "comet_qn_reseg_buckets_all_settings_all_systems.csv", index=False)


if __name__ == "__main__":
    run()


def reseg_buckets_all_settings(on_target: bool = True) -> pd.DataFrame:
    """The bucket boundary test of Section 7 on every Table 1 setting, one
    setting at a time (rating scales differ across corpora, so settings are
    not pooled with each other). The least fragmented bucket is the
    reference; QN cannot change agreement inside a bucket, only the pooled
    agreement across buckets."""
    from . import resegmentation
    rows = []
    for corpus, labels, setting, file, subset in resegmentation.SETTINGS:
        if corpus == "WMT25 CJK":
            continue          # six-bucket CJK ladder: not comparable with the eight-bucket test
        d = resegmentation.load_setting(file, subset, on_target)
        d = d[d.bucket.isin(LADDER8)]
        ref = np.sort(d[d.bucket == LADDER8[0]].score.to_numpy())
        within, parts = [], []
        for b in LADDER8:
            s = d[d.bucket == b]
            if len(s) < 10:
                continue
            q = qn_map(s.score, ref)
            within.append(abs(stats.spearman(s.score, s.human) - stats.spearman(q, s.human)))
            parts.append(pd.DataFrame({"raw": s.score.to_numpy(), "qn": q, "H": s.human.to_numpy()}))
        Q = pd.concat(parts, ignore_index=True)
        rows.append({"corpus": corpus, "labels": labels, "setting": setting, "n": len(Q),
                     "max_within_bucket_change": max(within),
                     "pooled_before": stats.spearman(Q.raw, Q.H),
                     "pooled_after": stats.spearman(Q.qn, Q.H)})
    out = pd.DataFrame(rows)
    out["pooled_gain"] = out.pooled_after - out.pooled_before
    return out

"""Sections 5.2, 5.3, 6 and 7 beyond the Indic languages.

Seven WMT settings in three non-Latin scripts (Devanagari: Hindi, Bhojpuri;
Cyrillic: Russian and Ukrainian in WMT24 and WMT25; Arabic), each translation
scored in its native script and romanised by a reversible letter-for-letter
scheme (experiments/data_prep/romanise_wmt.py). Human ratings are ESA, rated once on
the native text, as for IndicMT Eval.

Rows (point of comparison: native and romanised agreement are always computed
on the SAME rows): translations whose romanisation round-trips exactly and
that no metric truncates in EITHER condition. Each metric's own limit is read
from its code: COMET-22 encodes source, translation and reference separately
(each <= 510 XLM-R tokens); CometKiwi joins translation and source (<= 507,
comet/encoders/base.py concat_sequences); xCOMET also joins all three (<= 505);
MetricX-24 takes <= 1024 mT5 tokens; BLEURT-20 <= 512 for the pair; BERTScore
<= 512 mBERT tokens per side. Surface metrics do not truncate.

Two row sets: per metric (rows that metric does not truncate in either
condition; the main analysis) and common (rows no metric truncates; every
metric on identical rows). Exclusions are reported per condition.

Input: data/derived/beyond_indic/<pair>_<metric>.parquet (cluster output).
"""
from __future__ import annotations

import itertools
from functools import reduce

import numpy as np
import pandas as pd

from . import offtarget, stats
from .metaeval import ORIENT, acc_star_eq_pooled
from .paths import DATA, TABLES
from .qn import qn_map

B = DATA / "derived" / "beyond_indic"
PAIRS = {"en-hi": ("WMT24", "Devanagari", "Hindi"), "en-bho_IN": ("WMT25", "Devanagari", "Bhojpuri"),
         "en-ru": ("WMT24", "Cyrillic", "Russian"), "en-ru_RU": ("WMT25", "Cyrillic", "Russian"),
         "en-uk": ("WMT24", "Cyrillic", "Ukrainian"), "en-uk_UA": ("WMT25", "Cyrillic", "Ukrainian"),
         "en-ar_EG": ("WMT25", "Arabic", "Arabic")}
METRICS = ["comet22", "cometkiwi22", "cometkiwi23", "xcomet_xl", "xcomet_xxl",
           "metricx24", "bleurt20", "bertscore", "bleu", "chrf", "ter"]
NAMES = {"comet22": "COMET-22", "cometkiwi22": "CometKiwi-22", "cometkiwi23": "CometKiwi-23",
         "xcomet_xl": "xCOMET-XL", "xcomet_xxl": "xCOMET-XXL", "metricx24": "MetricX-24",
         "bleurt20": "BLEURT-20", "bertscore": "BERTScore", "bleu": "BLEU", "chrf": "chrF", "ter": "TER"}
CONDS = ["native", "romanised"]
KEY = ["system", "seg_idx"]
FILES = ["comet22", "cometkiwi22", "cometkiwi23", "xcomet_xl", "xcomet_xxl",
         "metricx24", "bleurt20", "bertscore", "surface"]


def fits(d: pd.DataFrame, metric: str, c: str) -> pd.Series | None:
    """Rows the metric scores without truncation in condition c; None if the
    token counts needed to decide are not available."""
    s, mt, rf = d.ntok_src, d[f"ntok_mt_{c}"], d[f"ntok_ref_{c}"]
    if metric == "comet22":
        return (s <= 510) & (mt <= 510) & (rf <= 510)
    if metric in ("cometkiwi22", "cometkiwi23"):
        return s + mt <= 507
    if metric in ("xcomet_xl", "xcomet_xxl"):
        return s + mt + rf <= 505
    need = {"metricx24": ["ntok_metricx"], "bleurt20": ["ntok_bleurt"],
            "bertscore": ["ntok_mbert_mt", "ntok_mbert_ref"]}.get(metric)
    if need is None:
        return pd.Series(True, index=d.index)
    if any(f"{n}_{c}" not in d for n in need):
        return None
    lim = 1024 if metric == "metricx24" else 512
    return reduce(lambda x, y: x & y, [d[f"{n}_{c}"] <= lim for n in need])


def load(pair: str, all_systems: bool = False) -> pd.DataFrame:
    """Every available score for one pair, all rows (filtering is per metric);
    off-target systems (tokinv.offtarget) are dropped unless all_systems."""
    frames = [pd.read_parquet(B / f"{pair}_tokens.parquet")]
    if (B / f"{pair}_tokens_g2.parquet").exists():
        frames.append(pd.read_parquet(B / f"{pair}_tokens_g2.parquet").drop(columns=["human", "roundtrip_ok"]))
    for m in FILES:
        f = B / f"{pair}_{m}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f).drop(columns=["human", "roundtrip_ok"]))
    d = reduce(lambda a, b: a.merge(b, on=KEY, how="inner", validate="1:1"), frames)
    return d if all_systems else d[~d.system.isin(offtarget.off_target(pair))].reset_index(drop=True)


def rows(d: pd.DataFrame, metric: str) -> tuple[pd.Series | None, dict]:
    """Mask of usable rows for a metric (None if undecidable) and the
    exclusion counts per condition."""
    fn, fr = fits(d, metric, "native"), fits(d, metric, "romanised")
    if fn is None or fr is None:
        return None, {}
    rt = d.roundtrip_ok
    keep = rt & fn & fr
    return keep, {"rows_scored": len(d), "rows_kept": int(keep.sum()),
                  "excl_roundtrip": int((~rt).sum()),
                  "excl_trunc_native_only": int((rt & ~fn & fr).sum()),
                  "excl_trunc_romanised_only": int((rt & fn & ~fr).sum()),
                  "excl_trunc_both": int((rt & ~fn & ~fr).sum())}


def common(d: pd.DataFrame) -> pd.Series | None:
    """Rows that no metric truncates in either condition."""
    masks = [rows(d, m)[0] for m in METRICS if f"{m}_native" in d]
    return None if any(m is None for m in masks) else reduce(lambda x, y: x & y, masks)


def usable(d: pd.DataFrame, metric: str) -> pd.DataFrame | None:
    """The metric's rows: scored, round-tripped, untruncated in both conditions."""
    if f"{metric}_native" not in d:
        return None
    keep, _ = rows(d, metric)
    return None if keep is None else d[keep].reset_index(drop=True)


def _agree(d, m, c):
    return ORIENT.get(m, 1) * stats.spearman(d[f"{m}_{c}"], d.human)


def reliability(rowset: str = "metric", ci: bool = True, all_systems: bool = False) -> pd.DataFrame:
    """Section 5.2: agreement with ESA per metric, native vs romanised, on the
    same rows, with a segment bootstrap (source segment = unit, all systems'
    translations of it resampled together) for both values and their difference."""
    out = []
    for pair, (corpus, script, lang) in PAIRS.items():
        d = load(pair, all_systems)
        cm = common(d) if rowset == "common" else None
        for m in METRICS:
            if f"{m}_native" not in d:
                continue
            keep, info = rows(d, m)
            if keep is None or (rowset == "common" and cm is None):
                continue
            if rowset == "common":
                keep = cm & d.roundtrip_ok
                info = {**info, "rows_kept": int(keep.sum())}
            x = d[keep].reset_index(drop=True)
            r = {"pair": pair, "corpus": corpus, "script": script, "language": lang,
                 "metric": NAMES[m], "rowset": rowset, "all_systems": all_systems,
                 "native": _agree(x, m, "native"), "romanised": _agree(x, m, "romanised")}
            r["diff"] = r["romanised"] - r["native"]
            if ci:
                for c in CONDS + ["diff"]:
                    f = ((lambda y, c=c: _agree(y, m, c)) if c != "diff" else
                         (lambda y: _agree(y, m, "romanised") - _agree(y, m, "native")))
                    r[f"{c}_lo"], r[f"{c}_hi"] = stats.segment_bootstrap(x, f, "seg_idx")
            # compared only if the native interval lies above zero
            r["compared"] = bool(r.get("native_lo", np.nan) > 0)
            out.append({**r, **info})
    return pd.DataFrame(out)


def qn_script(all_systems: bool = False) -> pd.DataFrame:
    """Section 7: native-to-romanised rescaling within each setting. The
    reference is the setting's own native distribution; agreement within each
    condition cannot change, pooled agreement over both conditions can."""
    rows = []
    for pair in PAIRS:
        full = load(pair, all_systems)
        for m in METRICS:
            d = usable(full, m)
            if d is None:
                continue
            sgn = ORIENT.get(m, 1)
            ref = np.sort(d[f"{m}_native"].to_numpy())
            raw = np.r_[d[f"{m}_native"], d[f"{m}_romanised"]] * sgn
            qn = np.r_[qn_map(d[f"{m}_native"], ref), qn_map(d[f"{m}_romanised"], ref)] * sgn
            h = np.r_[d.human, d.human]
            inv = max(abs(stats.spearman(d[f"{m}_{c}"], d.human)
                          - stats.spearman(qn_map(d[f"{m}_{c}"], ref), d.human)) for c in CONDS)
            rows.append({"pair": pair, "metric": NAMES[m], "pooled_before": stats.spearman(raw, h),
                         "pooled_after": stats.spearman(qn, h), "max_within_condition_change": inv})
    out = pd.DataFrame(rows)
    out["gain"] = out.pooled_after - out.pooled_before
    return out


def metaeval(all_systems: bool = False) -> pd.DataFrame:
    """Section 5.3: segment-level acc*_eq (pairs of systems on the same source)
    and system-ranking flips between the native and romanised scores."""
    rows = []
    for pair in PAIRS:
        full = load(pair, all_systems)
        for m in METRICS:
            d = usable(full, m)
            if d is None:
                continue
            sgn = ORIENT.get(m, 1)
            r = {"pair": pair, "metric": NAMES[m], "rows": len(d)}
            for c in CONDS:
                dh, dm = [], []
                for _, g in d.groupby("seg_idx"):
                    if len(g) < 2:
                        continue
                    i, j = np.triu_indices(len(g), 1)
                    h, s = g.human.to_numpy(float), g[f"{m}_{c}"].to_numpy(float) * sgn
                    dh.append(h[i] - h[j])
                    dm.append(s[i] - s[j])
                r[f"acc_star_eq_{c}"] = acc_star_eq_pooled(np.concatenate(dh), np.concatenate(dm))[0]
            means = {c: d.groupby("system")[f"{m}_{c}"].mean() * sgn for c in CONDS}
            systems = sorted(set(means["native"].index) & set(means["romanised"].index))
            pairs = list(itertools.combinations(systems, 2))
            r["n_systems"], r["n_pairs"] = len(systems), len(pairs)
            r["n_flips"] = int(sum(np.sign(means["native"][a] - means["native"][b])
                                   != np.sign(means["romanised"][a] - means["romanised"][b]) for a, b in pairs))
            rows.append(r)
    return pd.DataFrame(rows)


DOSE_CONDS = ["lossy", "noise02", "noise05", "noise10", "noise20"]


def dose(all_systems: bool = False) -> pd.DataFrame:
    """Section 6 beyond Indic: COMET-22 agreement against measured round-trip
    error. Lossless romanisation has error 0 by construction and exact round
    trips, so its drop from native is distortion with no information lost;
    the lossy scheme and the noise levels add known error on top. Rows: as
    the no-truncation rule, every condition untruncated (COMET-22: each side <= 510)."""
    out = []
    for pair, (corpus, script, lang) in PAIRS.items():
        f = B / f"{pair}_comet22_dose.parquet"
        if not f.exists():
            continue
        d = load(pair, all_systems)
        d = d.merge(pd.read_parquet(f).drop(columns=["human", "roundtrip_ok"]), on=KEY, validate="1:1")
        d = d.merge(pd.read_parquet(B / f"{pair}_dose_tokens.parquet"), on=KEY, validate="1:1")
        keep = d.roundtrip_ok & fits(d, "comet22", "native") & fits(d, "comet22", "romanised")
        for c in DOSE_CONDS:
            keep &= (d[f"ntok_mt_{c}"] <= 510) & (d[f"ntok_ref_{c}"] <= 510)
        x = d[keep].reset_index(drop=True)
        conds = [("native", None), ("lossless", "comet22_romanised")] + \
                [(c, f"comet22_{c}") for c in DOSE_CONDS]
        # does losing information (lossy) cost more agreement than the lossless
        # form that loses none? Paired on the same rows, segment bootstrap.
        f = lambda y: (stats.spearman(y.comet22_lossy, y.human)
                       - stats.spearman(y.comet22_romanised, y.human))
        lo, hi = stats.segment_bootstrap(x, f, "seg_idx")
        out.append({"pair": pair, "language": lang, "corpus": corpus,
                    "condition": "lossy_minus_lossless", "mean_cer": np.nan,
                    "agreement": f(x), "lo": lo, "hi": hi, "rows": len(x)})
        for c, col in conds:
            col = col or "comet22_native"
            cer = 0.0 if c in ("native", "lossless") else float(x[f"cer_{c}"].mean())
            lo, hi = stats.segment_bootstrap(x, lambda y, col=col: stats.spearman(y[col], y.human), "seg_idx")
            out.append({"pair": pair, "language": lang, "corpus": corpus, "condition": c,
                        "mean_cer": cer, "agreement": stats.spearman(x[col], x.human),
                        "lo": lo, "hi": hi, "rows": len(x)})
    return pd.DataFrame(out)


LEARNED = ["COMET-22", "CometKiwi-22", "CometKiwi-23", "xCOMET-XL", "xCOMET-XXL",
           "MetricX-24", "BLEURT-20", "BERTScore"]


def summary(rel: pd.DataFrame, me: pd.DataFrame) -> pd.DataFrame:
    """Every WMT number the main text prints, one row per (key, value):
    Table tab:wmt (COMET-22 per setting, learned-metric counts) and the
    pooled statements of Sections 5.2 and 5.3."""
    out = []
    add = lambda key, value, **k: out.append({"key": key, "value": float(value), **k})
    for pair, g in rel.groupby("pair", sort=False):
        c = g[g.metric == "COMET-22"].iloc[0]
        for col in ["native", "romanised", "diff", "diff_lo", "diff_hi", "romanised_lo", "romanised_hi"]:
            add(f"comet22_{col}", c[col], pair=pair)
        # how much romanisation lengthens the translations, in XLM-R tokens,
        # on the rows of the COMET-22 comparison
        full = load(pair)
        keep, _ = rows(full, "comet22")
        x = full[keep]
        add("tok_change_pct", 100 * (x.ntok_mt_romanised.mean() / x.ntok_mt_native.mean() - 1), pair=pair)
        lg = g[g.metric.isin(LEARNED) & g.compared]
        add("learned_compared", len(lg), pair=pair)
        add("learned_drop", int((lg.diff_hi < 0).sum()), pair=pair)
    lr = rel[rel.metric.isin(LEARNED)]
    add("learned_pairs_total", len(lr))
    add("learned_pairs_compared", int(lr.compared.sum()))
    add("learned_pairs_drop", int((lr.compared & (lr.diff_hi < 0)).sum()))
    add("learned_pairs_rise", int((lr.compared & (lr.diff_lo > 0)).sum()))
    add("bleu_max_abs_diff", rel[rel.metric == "BLEU"]["diff"].abs().max())
    m = me[me.compared]
    for metric, g in m.groupby("metric"):
        add("flips", g.n_flips.sum(), metric=metric)
        add("flip_pairs", g.n_pairs.sum(), metric=metric)
        add("flip_pct", 100 * g.n_flips.sum() / g.n_pairs.sum(), metric=metric)
    core = [x for x in LEARNED if x != "BERTScore"]
    pct = {x: 100 * m[m.metric == x].n_flips.sum() / m[m.metric == x].n_pairs.sum() for x in core}
    add("flip_pct_min_core", min(pct.values()))
    add("flip_pct_max_core", max(pct.values()))
    # the worked example of Section 5.2: a Russian translation rated 100
    ex = load("en-ru").set_index(KEY).loc[("Unbabel-Tower70B", 654)]
    for k, col in [("ex_human", "human"), ("ex_ntok_native", "ntok_mt_native"),
                   ("ex_ntok_romanised", "ntok_mt_romanised"), ("ex_comet22_native", "comet22_native"),
                   ("ex_comet22_romanised", "comet22_romanised")]:
        add(k, ex[col], pair="en-ru")
    return pd.DataFrame(out)


def run() -> None:
    for tag, every in [("", False), ("_all_systems", True)]:
        rel = reliability(all_systems=every)
        rel.to_csv(TABLES / f"beyond_indic_reliability{tag}.csv", index=False)
        flag = rel[["pair", "metric", "compared"]]
        qn_script(every).merge(flag, on=["pair", "metric"], how="left").to_csv(
            TABLES / f"beyond_indic_qn{tag}.csv", index=False)
        metaeval(every).merge(flag, on=["pair", "metric"], how="left").to_csv(
            TABLES / f"beyond_indic_metaeval{tag}.csv", index=False)
        dose(every).to_csv(TABLES / f"beyond_indic_dose{tag}.csv", index=False)
    reliability("common").to_csv(TABLES / "beyond_indic_reliability_common.csv", index=False)
    summary(pd.read_csv(TABLES / "beyond_indic_reliability.csv"),
            pd.read_csv(TABLES / "beyond_indic_metaeval.csv")).to_csv(
        TABLES / "beyond_indic_summary.csv", index=False)


if __name__ == "__main__":
    run()

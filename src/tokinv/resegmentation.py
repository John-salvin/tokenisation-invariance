"""Section 4 and App. Per-Bucket Results: byte-identical re-segmentation.

Each segment is re-tokenised into non-canonical XLM-R segmentations that decode
to the identical byte string, binned by fragmentation ratio (tokens / canonical
tokens) into the eight-bucket ladder, and re-scored. The question is whether
metric-human agreement falls as fragmentation rises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import offtarget, stats
from .paths import INDIC_LANGS, LADDER8, TABLES
from .paths import DATA

SCORES = DATA / "scores" / "resegmentation"

# The twenty settings of Table 1, in the paper's order, plus the three CJK
# settings of Section 4.2 and the vocabulary appendix. `file` is the
# per-segment score file in data/scores/resegmentation/. Every WMT setting
# uses a 1,200-segment rerun: `hp` for the WMT24 pairs, three WMT25 pairs and
# CJK (experiments/tokenisation/wmt_reseg_1200.sbatch), `pw` for the
# other five WMT25 pairs (wmt_reseg_1200_wmt25.sbatch).
SETTINGS = (
    [("IndicMT Eval", "MQM", lang.upper(), "indic_comet22", lang) for lang in INDIC_LANGS]
    + [("WMT24", "MQM", "SPA", "latin_wmt24_comet22", "spa"),
       ("WMT24", "MQM", "DEU", "latin_wmt24_comet22", "deu")]
    + [("WMT24", "ESA", p, f"wmt_{p}_hp", None)
       for p in ["en-cs", "en-uk", "en-ru", "en-is", "en-hi"]]
    + [("WMT25", "ESA", "en-cs_CZ", "wmt_en-cs_CZ_hp", None),
       ("WMT25", "ESA", "en-is_IS", "wmt_en-is_IS_pw", None),
       ("WMT25", "ESA", "en-bho_IN", "wmt_en-bho_IN_hp", None),
       ("WMT25", "ESA", "en-uk_UA", "wmt_en-uk_UA_pw", None),
       ("WMT25", "ESA", "en-ar_EG", "wmt_en-ar_EG_hp", None),
       ("WMT25", "ESA", "en-ru_RU", "wmt_en-ru_RU_pw", None),
       ("WMT25", "ESA", "en-et_EE", "wmt_en-et_EE_pw", None),
       # WMT25 provides no Italian reference (refA is the string NaN), so the
       # reference-based COMET-22 run is degenerate; reference-free CometKiwi-22
       # on the same rows replaces it (wmt_en-it_IT_pw kept for the record)
       ("WMT25", "ESA", "en-it_IT", "wmt_en-it_IT_kiwi22_pw", None)]
    + [("WMT25 CJK", "ESA", "en-ja_JP", "wmt_en-ja_JP_hp_cjk", None),
       ("WMT25 CJK", "ESA", "en-zh_CN", "wmt_en-zh_CN_hp_cjk", None),
       ("WMT25 CJK", "MQM", "en-ko_KR", "wmt_en-ko_KR_hp_cjk", None)]
)


def load_setting(file: str, subset: str | None, on_target: bool = True) -> pd.DataFrame:
    """Per-segment scores in a common schema: unit, bucket, ratio, score, human.
    on_target drops the setting's off-target systems (WMT only)."""
    d = pd.read_parquet(SCORES / f"{file}.parquet")
    if on_target and file.startswith("wmt_"):
        setting = (file[len("wmt_"):].split("_hp")[0].split("_pw")[0].split("_base")[0]
                   .split("_cjk")[0].split("_kiwi22")[0])
        d = d[~d.system.isin(offtarget.off_target(setting))]
    if file == "indic_comet22":
        d = d[(d.lang == subset) & (d.direction == "forward") & d.bucket.isin(LADDER8)]
        return d.rename(columns={"segment_id": "unit", "comet22": "score",
                                 "mqm_score": "human"})
    if file == "latin_wmt24_comet22":
        d = d[d.pair == subset]
        return d.rename(columns={"seg_id": "unit", "comet22": "score",
                                 "mqm_score": "human"})
    return d.rename(columns={"seg_idx": "unit", "comet22": "score"})


def all_settings(n_resamples: int = stats.BOOT_RESAMPLES,
                 seed: int = stats.BOOT_SEED, on_target: bool = True) -> pd.DataFrame:
    """Table 1 / App. Per-Bucket Results: monotonicity rho with its segment-bootstrap interval."""
    rows = []
    for corpus, labels, setting, file, subset in SETTINGS:
        d = load_setting(file, subset, on_target)
        rho = stats.monotonicity(d)
        lo, hi, draws = stats.segment_bootstrap(d, stats.monotonicity, "unit",
                                                n_resamples=n_resamples, seed=seed,
                                                return_draws=True)
        rows.append({"corpus": corpus, "labels": labels, "setting": setting,
                     "rho": round(rho, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                     "excl0": bool(hi < 0 or lo > 0), "n_segments": d.unit.nunique(),
                     # share of resamples on the far side of zero: how close
                     # a setting sits to the interval's 2.5 percent boundary
                     "share_rho_ge0": float((draws >= 0).mean()),
                     "rho_3dp": round(rho, 3)})
    return pd.DataFrame(rows)


# target language of each Table 1 setting (ISO 639-3), for counting languages
LANGUAGE = {"GUJ": "guj", "TAM": "tam", "MAL": "mal", "MAR": "mar", "HIN": "hin", "SPA": "spa", "DEU": "deu",
            "en-cs": "ces", "en-uk": "ukr", "en-ru": "rus", "en-is": "isl", "en-hi": "hin", "en-cs_CZ": "ces",
            "en-is_IS": "isl", "en-bho_IN": "bho", "en-uk_UA": "ukr", "en-ar_EG": "arz", "en-ru_RU": "rus",
            "en-et_EE": "est", "en-it_IT": "ita"}


def settings_summary() -> pd.DataFrame:
    """Counts the abstract and Section 4 print: settings and distinct target
    languages in Table 1."""
    full = [s for s in SETTINGS if s[0] != "WMT25 CJK"]
    langs = {LANGUAGE[s[2]] for s in full}
    return pd.DataFrame([{"n_settings": len(full), "n_languages": len(langs)}])


def monotonicity_exact_p() -> pd.DataFrame:
    """Exact two-sided permutation p of the monotonicity rho, Indic and Latin."""
    rows = []
    for _, _, setting, file, subset in SETTINGS[:7]:
        c = stats.dose_curve(load_setting(file, subset))
        rho = stats.spearman(c.mean_ratio, c.rho)
        rows.append({"lang": subset, "rho": round(rho, 3),
                     "exact_p": round(stats.exact_p(rho, len(c)), 5)})
    return pd.DataFrame(rows)


def dose_response() -> pd.DataFrame:
    """Per-language, per-bucket agreement on IndicMT Eval (Table J.2)."""
    d = pd.read_parquet(SCORES / "indic_comet22.parquet")
    rows = []
    for lang in INDIC_LANGS:
        for direction in ["forward", "reverse"]:
            sub_dir = d[(d.lang == lang) & (d.direction == direction)]
            for bucket in sub_dir.bucket.unique():
                s = sub_dir[sub_dir.bucket == bucket]
                if len(s) < 4:
                    continue
                rho = (stats.spearman(s.comet22, s.mqm_score)
                       if s.comet22.nunique() > 1 else float("nan"))
                rows.append({"lang": lang, "direction": direction, "bucket": bucket,
                             "n": len(s), "mean_ratio": s.ratio.mean(),
                             "mean_comet22": s.comet22.mean(),
                             "spearman_comet_mqm": rho})
    return pd.DataFrame(rows)


def bucket_reachability() -> pd.DataFrame:
    """Share of (segment, seed) draws for which the sampler reached each bucket."""
    s = pd.read_parquet(SCORES / "indic_samples_meta.parquet")
    g = (s.groupby(["lang", "direction", "bucket"], sort=True)
          .agg(n_total=("reachable", "size"), n_reachable=("reachable", "sum"))
          .reset_index())
    g["pct_reachable"] = 100.0 * g.n_reachable / g.n_total
    return g


def latin_buckets() -> pd.DataFrame:
    """WMT24 German and Spanish (MQM): agreement per bucket, canonical first."""
    d = pd.read_parquet(SCORES / "latin_wmt24_comet22.parquet")
    rows = []
    for pair in ["deu", "spa"]:
        p = d[d.pair == pair]
        for b in ["canonical"] + LADDER8:
            s = p[p.bucket == b]
            if len(s) < stats.MIN_BUCKET_N:
                continue
            rows.append({"pair": pair, "bucket": b, "n": len(s),
                         "mean_ratio": float(s.ratio.mean()),
                         "rho_vs_mqm": stats.spearman(s.comet22, s.mqm_score),
                         "mean_comet": float(s.comet22.mean())})
    return pd.DataFrame(rows)


def latin_summary(buckets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, p in buckets.groupby("pair", sort=True):
        can = float(p.loc[p.bucket == "canonical", "rho_vs_mqm"].iloc[0])
        nc = p[p.bucket != "canonical"]
        last = float(nc.rho_vs_mqm.iloc[-1])
        rows.append({"pair": pair, "canonical_rho": can, "most_fragmented_rho": last,
                     "absolute_drop": can - last, "relative_drop": (can - last) / can,
                     "monotonicity_spearman": stats.spearman(nc.mean_ratio, nc.rho_vs_mqm),
                     "n_buckets": len(nc)})
    return pd.DataFrame(rows)


# Length filter of the Latin controls: the most fragmented (3x) form of a
# 160-token target still fits XLM-R's 512 positions; source and reference must
# fit as they are. Rows are dropped, never truncated, because truncation would
# break byte identity.
MAX_CANONICAL_TOKENS = 160
MAX_SEQ_TOKENS = 510


def latin_retained_rows() -> pd.DataFrame:
    from . import tokenisers as T
    w = pd.read_parquet(DATA / "wmt" / "wmt24_mqm_en-de_en-es.parquet")
    rows = []
    for pair in ["deu", "spa"]:
        d = w[w.pair == pair].dropna(subset=["source", "target", "refA", "mqm_score"])
        d = d.drop_duplicates(subset=["system", "seg_id"])
        ck = d.target.map(lambda x: T.n_tokens("xlmr", str(x).replace(T.NBSP, " ")))
        sk = d.source.map(lambda x: T.n_tokens("xlmr", str(x)))
        rk = d.refA.map(lambda x: T.n_tokens("xlmr", str(x)))
        keep = (ck <= MAX_CANONICAL_TOKENS) & (sk <= MAX_SEQ_TOKENS) & (rk <= MAX_SEQ_TOKENS)
        rows.append({"pair": pair, "rows_total": len(d), "rows_kept": int(keep.sum()),
                     "systems": int(d.system.nunique()), "segments": int(d.seg_id.nunique())})
    return pd.DataFrame(rows)


def mt5_dose_response() -> pd.DataFrame:
    """The same ladder on MetricX-24, re-segmenting with mT5's vocabulary.
    MetricX is an error score (lower is better), so agreement is |rho|."""
    d = pd.read_parquet(SCORES / "indic_metricx24_n300.parquet")
    rows = []
    for lang in INDIC_LANGS:
        sub_dir = d[(d.lang == lang) & (d.direction == "forward")]
        for bucket in sub_dir.bucket.unique():
            s = sub_dir[sub_dir.bucket == bucket]
            if len(s) < 4:
                continue
            rho = stats.spearman(s.metricx24, s.mqm_score)
            rows.append({"lang": lang, "direction": "forward", "bucket": bucket,
                         "n": len(s), "mean_ratio": s.ratio.mean(),
                         "mean_metricx24": s.metricx24.mean(),
                         "spearman_metricx_mqm": rho, "abs_spearman_metricx_mqm": abs(rho)})
    return pd.DataFrame(rows)


def mt5_significance(curve: pd.DataFrame) -> pd.DataFrame:
    """Monotonicity of |rho| over the buckets mT5 can reach. Gujarati reaches
    seven of the eight, so its exact p uses n = 7."""
    rows = []
    for lang in ["guj", "hin", "mal", "tam", "mar"]:
        c = curve[(curve.lang == lang) & curve.bucket.isin(LADDER8)]
        rho = stats.spearman(c.mean_ratio, c.abs_spearman_metricx_mqm)
        p = stats.exact_p(rho, len(c))
        rows.append({"lang": lang, "spearman_ratio_vs_abs_rho": rho, "n_buckets": len(c),
                     "exact_permutation_p": round(p, 4), "significant_at_05": p < 0.05})
    return pd.DataFrame(rows)


def wmt_top_bucket() -> pd.DataFrame:
    """Agreement in the most fragmented bucket, per WMT setting and run: the
    Table 1 run, plus the first, smaller runs (`base`, `cjk`) of Arabic,
    Bhojpuri and CJK for comparison. The paper prints the Table 1 run."""
    rows = []
    for _, _, setting, file, subset in SETTINGS[7:]:
        runs = [(file.replace(f"wmt_{setting}_", ""), file)]
        if setting in ("en-ar_EG", "en-bho_IN"):
            runs.append(("base", f"wmt_{setting}_base"))
        if setting in ("en-ja_JP", "en-zh_CN", "en-ko_KR"):
            runs.append(("cjk", f"wmt_{setting}_cjk"))
        for run, f in runs:
            d = load_setting(f, subset)
            nc = d[d.bucket != "canonical"]
            last = nc[nc.bucket == sorted(nc.bucket.unique())[-1]]
            rows.append({"setting": setting, "run": run, "n_segments": d.unit.nunique(),
                         "top_bucket": last.bucket.iloc[0],
                         "n": len(last), "rho_top_bucket": stats.spearman(last.score, last.human)})
    return pd.DataFrame(rows)


def mt5_n900_bootstrap() -> pd.DataFrame:
    """The MetricX-24 replication rerun at 900 segments per language, with the
    standard segment bootstrap. MetricX is an error score, so a falling |rho|
    reads as a POSITIVE monotonicity here."""
    d = pd.read_parquet(SCORES / "indic_metricx24_n900.parquet")
    d = d[(d.direction == "forward") & d.bucket.isin(LADDER8)].rename(
        columns={"segment_id": "unit", "metricx24": "score", "mqm_score": "human"})
    rows = []
    for lang in INDIC_LANGS:
        x = d[d.lang == lang]
        lo, hi = stats.segment_bootstrap(x, stats.monotonicity, "unit")
        rows.append({"lang": lang, "rho": round(stats.monotonicity(x), 3),
                     "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "excl0": bool(hi < 0 or lo > 0)})
    return pd.DataFrame(rows)


def run() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    dose_response().to_csv(TABLES / "reseg_dose_response.csv", index=False)
    bucket_reachability().to_csv(TABLES / "reseg_bucket_reachability.csv", index=False)
    monotonicity_exact_p().to_csv(TABLES / "reseg_monotonicity_exact_p.csv", index=False)
    lb = latin_buckets()
    lb.to_csv(TABLES / "latin_wmt24_reseg_buckets.csv", index=False)
    latin_summary(lb).to_csv(TABLES / "latin_wmt24_reseg_summary.csv", index=False)
    latin_retained_rows().to_csv(TABLES / "latin_wmt24_retained_rows.csv", index=False)
    mc = mt5_dose_response()
    mc.to_csv(TABLES / "reseg_dose_response_mt5.csv", index=False)
    mt5_significance(mc).to_csv(TABLES / "reseg_mt5_significance.csv", index=False)
    wmt_top_bucket().to_csv(TABLES / "reseg_wmt_top_bucket.csv", index=False)
    mt5_n900_bootstrap().to_csv(TABLES / "metricx_n900_bootstrap.csv", index=False)
    all_settings().to_csv(TABLES / "reseg_all_settings.csv", index=False)
    all_settings(on_target=False).to_csv(TABLES / "reseg_all_settings_all_systems.csv", index=False)
    settings_summary().to_csv(TABLES / "reseg_settings_summary.csv", index=False)


if __name__ == "__main__":
    run()

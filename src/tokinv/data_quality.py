"""Appendix "Data Quality Findings" and the two worked examples.

Everything here is computed from data/indic/segments.parquet with the pinned
XLM-R tokeniser; nothing needs a GPU.
"""
from __future__ import annotations

import re

import pandas as pd

from . import sampler as S
from . import tokenisers as T
from .paths import INDIC, INDIC_LANGS, TABLES

ZWJ, ZWNJ = "‍", "‌"
# a space before closing punctuation: the romanised condition inserts one,
# which blocks positional word alignment until normalised
STRAY_SPACE = re.compile(r"\s+[.,!?;:)\]}%'\"।॥]")


def _segments() -> pd.DataFrame:
    seg = pd.read_parquet(INDIC / "segments.parquet")
    seg["lang"] = seg["lang"].astype(str)
    seg["condition"] = seg["condition"].astype(str)
    return seg


def nbsp_prevalence(seg) -> pd.DataFrame:
    """Share of targets containing U+00A0. SentencePiece decodes every word
    boundary as a plain space, so an NBSP segment can never round-trip under
    any segmentation; the sampler maps NBSP to space first."""
    g = seg.groupby(["lang", "condition"], sort=False).target
    out = g.apply(lambda s: s.astype(str).str.contains(S.NBSP).mean()).reset_index(name="nbsp_prevalence")
    order = {l: i for i, l in enumerate(INDIC_LANGS)}
    return out.sort_values(["lang", "condition"], key=lambda c: c.map(order) if c.name == "lang" else c).reset_index(drop=True)


def zwj_reachability(seg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Native segments the sampler must exclude because the tokeniser's OWN
    canonical tokenisation does not round-trip (it drops zero-width joiners),
    and whether the excluded segments differ in quality."""
    nat = seg[seg.mqm_valid & (seg.condition == "native")]
    rows = []
    for r in nat.itertuples():
        text = str(r.target)
        try:
            S.build_mdds(text, "xlmr")
            status = "retained"
        except S.CanonicalRoundtripFailure:
            status = "excluded_roundtrip_failure"
        except S.NoncanonicalError as exc:
            status = f"excluded_other:{type(exc).__name__}"
        rows.append({"lang": r.lang, "segment_id": r.segment_id, "comet_22": r.comet_22,
                     "status": status, "n_zwj": text.count(ZWJ), "n_zwnj": text.count(ZWNJ),
                     "has_any_zwj_zwnj": (ZWJ in text) or (ZWNJ in text),
                     "n_words": len(text.split())})
    full = pd.DataFrame(rows)
    summ = []
    for lang in INDIC_LANGS:
        sub = full[full.lang == lang]
        exc, ret = sub[sub.status != "retained"], sub[sub.status == "retained"]
        nan = float("nan")
        summ.append({"lang": lang, "n_total": len(sub), "n_excluded": len(exc),
                     "pct_excluded": 100 * len(exc) / len(sub), "n_retained": len(ret),
                     "pct_excluded_have_zwj_zwnj": 100 * exc.has_any_zwj_zwnj.mean() if len(exc) else nan,
                     "pct_retained_have_zwj_zwnj": 100 * ret.has_any_zwj_zwnj.mean() if len(ret) else nan,
                     "mean_comet22_excluded": exc.comet_22.mean() if len(exc) else nan,
                     "mean_comet22_retained": ret.comet_22.mean() if len(ret) else nan,
                     "comet22_gap_retained_minus_excluded":
                         ret.comet_22.mean() - exc.comet_22.mean() if len(exc) and len(ret) else nan})
    return full, pd.DataFrame(summ)


def stray_space(seg) -> pd.DataFrame:
    rows = []
    for lang in ["guj", "hin", "mal", "mar", "tam"]:
        for cond in ["native", "romanised"]:
            s = seg[(seg.lang == lang) & (seg.condition == cond)].target.dropna().astype(str)
            rows.append({"lang": lang, "condition": cond, "n": len(s),
                         "pct_with_stray_space":
                             round(100 * float(s.map(lambda x: bool(STRAY_SPACE.search(x))).mean()), 1)})
    return pd.DataFrame(rows)


def n_segmentations(word: str) -> int:
    """Vocabulary-valid segmentations of one word-initial whitespace word."""
    vocab = T.get("xlmr").get_vocab()
    s = "▁" + word
    memo = {len(s): 1}

    def count(i):
        if i not in memo:
            memo[i] = sum(count(j) for j in range(len(s), i, -1) if s[i:j] in vocab)
        return memo[i]
    return count(0)


def worked_examples(seg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Example 1: the Gujarati segment with the maximum MQM whose COMET-22
    falls furthest under romanisation. Example 2: segmentation counts."""
    g = seg[seg.lang == "guj"]
    nat = g[g.condition == "native"].set_index("segment_id")
    rom = g[g.condition == "romanised"].set_index("segment_id")
    d = nat[["comet_22", "mqm_score", "tok_count_xlmr", "system"]].join(
        rom[["comet_22", "tok_count_xlmr"]], rsuffix="_rom").dropna(subset=["comet_22", "comet_22_rom", "mqm_score"])
    top = d[d.mqm_score >= d.mqm_score.max()]
    r = top.assign(drop=top.comet_22 - top.comet_22_rom).sort_values("drop", ascending=False).iloc[0]
    seg_ex = pd.DataFrame([{"mqm": float(r.mqm_score), "mqm_max": float(d.mqm_score.max()),
                            "comet_native": round(float(r.comet_22), 1),
                            "comet_romanised": round(float(r.comet_22_rom), 1),
                            "comet_drop": round(float(r.comet_22) - float(r.comet_22_rom), 0),
                            "ntok_native": int(r.tok_count_xlmr),
                            "ntok_romanised": int(r.tok_count_xlmr_rom), "system": r.system}])
    tok = T.get("xlmr")
    words = pd.DataFrame([{"word": w, "canonical_tokens": len(tok.tokenize(w)),
                           "n_segmentations": n_segmentations(w)}
                          for w in ["jaheratanum", "announcement",
                                    "જાહેરાતનું"]])
    return seg_ex, words


def run() -> None:
    seg = _segments()
    nbsp_prevalence(seg).to_csv(TABLES / "reseg_nbsp_prevalence.csv", index=False)
    full, summ = zwj_reachability(seg)
    full.to_csv(TABLES / "zwj_reachability_all_langs.csv", index=False)
    summ.to_csv(TABLES / "zwj_reachability_summary.csv", index=False)
    stray_space(seg).to_csv(TABLES / "stray_space_prevalence.csv", index=False)
    ex, words = worked_examples(seg)
    ex.to_csv(TABLES / "worked_example_segment.csv", index=False)
    words.to_csv(TABLES / "worked_example_words.csv", index=False)


if __name__ == "__main__":
    run()

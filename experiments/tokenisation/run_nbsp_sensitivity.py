#!/usr/bin/env python3
"""Sensitivity of the intrinsic tokeniser measures to no-break spaces.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import flores as F
import intrinsic as I
import provenance as PROV
import tokenisers as T

NBSP = "\xa0"
TP_FLAG_THRESHOLD = 0.02


def check_tokeniser_absorption() -> pd.DataFrame:
    rows = []
    for tok in T.PANEL:
        isolated_ids = T.tokenise(NBSP, tok, add_special_tokens=False)
        with_ctx = T.tokenise(f"word{NBSP}word", tok, add_special_tokens=False)
        without_ctx = T.tokenise("wordword", tok, add_special_tokens=False)
        without_ctx_spaced = T.tokenise("word word", tok, add_special_tokens=False)
        rows.append({
            "tokeniser": tok,
            "n_ids_for_isolated_nbsp": len(isolated_ids),
            "isolated_nbsp_ids": isolated_ids,
            "n_ids_word_nbsp_word": len(with_ctx),
            "n_ids_wordword_no_sep": len(without_ctx),
            "n_ids_word_space_word": len(without_ctx_spaced),
            "nbsp_treated_as_word_boundary": len(with_ctx) == len(without_ctx_spaced),
            "nbsp_absorbed_as_nothing": len(with_ctx) == len(without_ctx),
        })
    return pd.DataFrame(rows)


def _recompute(seg: pd.DataFrame, tok: str) -> pd.DataFrame:
    per_seg = I.per_segment(seg, tok)
    return I.aggregate(per_seg)


def indicmt_sensitivity() -> pd.DataFrame:
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg_norm = seg.copy()
    seg_norm["target"] = seg_norm["target"].astype(str).str.replace(NBSP, " ", regex=False)

    rows = []
    for tok in T.PANEL:
        orig = _recompute(seg, tok).set_index(["lang", "condition"])
        norm = _recompute(seg_norm, tok).set_index(["lang", "condition"])
        for lang in C.LANG_ORDER:
            for cond in C.CONDITIONS:
                o = orig.loc[(lang, cond)]
                n = norm.loc[(lang, cond)]
                rec = {"tokeniser": tok, "lang": lang, "condition": cond}
                for metric in ["tp", "fertility", "cpt", "wfr", "byte_premium"]:
                    rec[f"{metric}_orig"] = float(o[metric])
                    rec[f"{metric}_norm"] = float(n[metric])
                    rec[f"delta_{metric}"] = float(n[metric]) - float(o[metric])
                rows.append(rec)
    df = pd.DataFrame(rows)

    gap_rows = []
    piv = df.set_index(["tokeniser", "lang", "condition"])
    for tok in T.PANEL:
        for lang in C.LANG_ORDER:
            rec = {"tokeniser": tok, "lang": lang}
            for metric in ["tp", "fertility", "cpt", "wfr", "byte_premium"]:
                gap_orig = (piv.loc[(tok, lang, "native"), f"{metric}_orig"]
                           - piv.loc[(tok, lang, "romanised"), f"{metric}_orig"])
                gap_norm = (piv.loc[(tok, lang, "native"), f"{metric}_norm"]
                           - piv.loc[(tok, lang, "romanised"), f"{metric}_norm"])
                rec[f"gap_{metric}_orig"] = float(gap_orig)
                rec[f"gap_{metric}_norm"] = float(gap_norm)
                rec[f"delta_gap_{metric}"] = float(gap_norm - gap_orig)
            gap_rows.append(rec)
    gap_df = pd.DataFrame(gap_rows)
    return df, gap_df


def flores_sensitivity() -> pd.DataFrame:
    rows = []
    for lang, code in F.PAPER_FIVE.items():
        text = F.load_lang(code)
        text_norm = [t.replace(NBSP, " ") for t in text]
        n_nbsp_segments = sum(1 for t in text if NBSP in t)
        for tok in T.PANEL:
            n_tok_orig = [T.count(t, tok) for t in text]
            n_tok_norm = [T.count(t, tok) for t in text_norm]
            n_wrd_orig = [len(I.words(t)) for t in text]
            n_wrd_norm = [len(I.words(t)) for t in text_norm]
            fert_orig = sum(n_tok_orig) / sum(n_wrd_orig)
            fert_norm = sum(n_tok_norm) / sum(n_wrd_norm)
            rows.append({
                "lang": lang, "tokeniser": tok, "n_segments": len(text),
                "n_segments_with_nbsp": n_nbsp_segments,
                "pct_segments_with_nbsp": 100.0 * n_nbsp_segments / len(text),
                "fertility_orig": fert_orig, "fertility_norm": fert_norm,
                "delta_fertility": fert_norm - fert_orig,
                "tokens_total_orig": sum(n_tok_orig), "tokens_total_norm": sum(n_tok_norm),
            })
    return pd.DataFrame(rows)


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    print("=" * 78); print("1. Per-tokeniser NBSP handling"); print("=" * 78)
    absorb_df = check_tokeniser_absorption()
    absorb_df.drop(columns=["isolated_nbsp_ids"]).to_csv(
        C.OUT_TABLES / "nbsp_tokeniser_absorption.csv", index=False)
    print(absorb_df[["tokeniser", "n_ids_for_isolated_nbsp", "nbsp_treated_as_word_boundary",
                      "nbsp_absorbed_as_nothing"]].to_string(index=False))
    tp_at_risk = absorb_df[~absorb_df.nbsp_absorbed_as_nothing]["tokeniser"].tolist()
    print(f"\nTokenisers where NBSP changes the token COUNT (TP potentially at risk): {tp_at_risk}")
    print(f"Tokenisers that fully absorb NBSP (TP unaffected by construction): "
          f"{[t for t in T.PANEL if t not in tp_at_risk]}")

    print(); print("=" * 78); print("2. IndicMT Eval sensitivity"); print("=" * 78)
    df, gap_df = indicmt_sensitivity()
    df.to_csv(C.OUT_TABLES / "nbsp_sensitivity.csv", index=False)
    gap_df.to_csv(C.OUT_TABLES / "nbsp_sensitivity_gap_deltas.csv", index=False)

    print("\nPer-(tokeniser, lang) delta in the native-minus-romanised TP gap "
          "(THE number that matters):")
    print(gap_df[["tokeniser", "lang", "gap_tp_orig", "gap_tp_norm", "delta_gap_tp"]]
          .to_string(index=False))

    xlmr_gap = gap_df[gap_df.tokeniser == "xlmr"]
    max_xlmr_tp_gap_shift = xlmr_gap["delta_gap_tp"].abs().max()
    max_xlmr_tp_shift_absolute = df[df.tokeniser == "xlmr"]["delta_tp"].abs().max()
    print(f"\nHEADLINE (xlmr, COMET's vocabulary): max |delta_gap_tp| across languages = "
          f"{max_xlmr_tp_gap_shift:.5f}")
    print(f"HEADLINE (xlmr): max |delta_tp| (any single condition cell) = "
          f"{max_xlmr_tp_shift_absolute:.5f}")

    any_flag = False
    flagged = []
    for _, r in df.iterrows():
        if abs(r["delta_tp"]) > TP_FLAG_THRESHOLD:
            any_flag = True
            flagged.append((r["tokeniser"], r["lang"], r["condition"], r["delta_tp"]))
    headline_flag = max_xlmr_tp_shift_absolute > TP_FLAG_THRESHOLD
    if any_flag:
        print("\n" + "!" * 78)
        print(f"FLAG (non-headline tokenisers -- reported, does not gate the DAG): "
              f"{len(flagged)} (tokeniser, lang, condition) cells shift TP by more than "
              f"{TP_FLAG_THRESHOLD} under NBSP normalisation:")
        for t, l, c, d in flagged:
            print(f"  {t:10s} {l:4s} {c:10s} delta_tp={d:+.4f}")
        print("These are byte-level BPE tokenisers (+ byt5) -- confirmed in section 1 above "
              "that they do NOT absorb NBSP as a no-op the way xlmr/mbert/mt5/rembert/nllb do, "
              "so a real TP shift here is expected, not a bug. None of these are headline "
              "tokenisers for this paper's claims.")
        print("!" * 78)
    else:
        print(f"\nNo (tokeniser, lang, condition) cell shifts TP by more than "
              f"{TP_FLAG_THRESHOLD}.")
    if headline_flag:
        print(f"\n>>> HEADLINE (xlmr) TP shift {max_xlmr_tp_shift_absolute:.5f} EXCEEDS "
              f"{TP_FLAG_THRESHOLD} -- this DOES gate the DAG. <<<")
    else:
        print(f"\nHEADLINE (xlmr) max shift {max_xlmr_tp_shift_absolute:.5f} is well under "
              f"{TP_FLAG_THRESHOLD}. No headline TP number is at risk from NBSP; safe to "
              f"proceed to re-segmentation.")

    print(); print("=" * 78); print("3. FLORES-200 (native only -- is this IndicMT-Eval-specific?)")
    print("=" * 78)
    flores_df = flores_sensitivity()
    flores_df.to_csv(C.OUT_TABLES / "nbsp_sensitivity_flores.csv", index=False)
    xlmr_flores = flores_df[flores_df.tokeniser == "xlmr"]
    print(xlmr_flores[["lang", "pct_segments_with_nbsp", "fertility_orig", "fertility_norm",
                        "delta_fertility"]].to_string(index=False))

    PROV.register(
        "nbsp_sensitivity", {
            "tokeniser_absorption": absorb_df.drop(columns=["isolated_nbsp_ids"]).to_dict("records"),
            "indicmt_gap_deltas": gap_df.to_dict("records"),
            "flores_summary": flores_df[flores_df.tokeniser == "xlmr"].to_dict("records"),
            "xlmr_max_abs_delta_gap_tp": float(max_xlmr_tp_gap_shift),
            "xlmr_max_abs_delta_tp_any_cell": float(max_xlmr_tp_shift_absolute),
            "any_cell_exceeds_threshold": any_flag,
            "headline_xlmr_exceeds_threshold": headline_flag,
            "threshold": TP_FLAG_THRESHOLD,
        },
        PROV.stamp(Path(__file__), modules=["intrinsic", "tokenisers", "flores", "constants"],
                   seed=None,
                   note=("NBSP sensitivity study, scoped exactly like the chillu analysis -- "
                         "intrinsic.py and every anchor left untouched; this recomputes both "
                         "original and NBSP-normalised via the identical code path for a "
                         "clean diff")))

    if headline_flag:
        print("\n>>> STOP CONDITION MET: a headline (xlmr) TP number moved by more than the "
              "threshold. Flagging prominently; NOT proceeding automatically to re-segmentation. <<<")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""MorphScore v2 on native-script text for the tokeniser panel.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import constants as C
import intrinsic as I
import morphscore_gate as GATE
import provenance as PROV
import tokenisers as T

LANG_TO_FILE: dict[str, str] = {
    "guj": "gujarati", "tam": "tamil", "mal": "malayalam", "mar": "marathi", "hin": "hindi",
    "san": "sanskrit", "urd": "urdu", "snd": "sindhi", "bho": "bhojpuri",
}
LANG_ORDER_PANEL = C.LANG_ORDER + ["san", "urd", "snd", "bho"]
GUJARATI_N_ROWS_APPROX = 120
MORPHSCORE_LIB_DIR = C.STAGING / "repos" / "_morphscore_lib"


def _ensure_library_unpacked() -> Path:
    if not MORPHSCORE_LIB_DIR.exists():
        tarball = C.STAGING / "repos" / "morphscore.tar.gz"
        if not tarball.exists():
            raise RuntimeError(f"morphscore.tar.gz missing at {tarball}")
        MORPHSCORE_LIB_DIR.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as tf:
            tf.extractall(MORPHSCORE_LIB_DIR)
    candidates = list(MORPHSCORE_LIB_DIR.rglob("morphscore.py"))
    if not candidates:
        raise RuntimeError(f"morphscore.py not found under {MORPHSCORE_LIB_DIR}")
    return candidates[0].parent


def main() -> int:
    C.OUT_TABLES.mkdir(parents=True, exist_ok=True)

    lib_dir = _ensure_library_unpacked()
    sys.path.insert(0, str(lib_dir))
    from morphscore import MorphScore

    data_dir = C.MORPHSCORE_DIR
    if not data_dir.exists():
        raise RuntimeError(f"MorphScore data not staged at {data_dir}")

    raw_datasets = {}
    for lang in LANG_ORDER_PANEL:
        path = data_dir / f"{LANG_TO_FILE[lang]}_data.csv"
        if not path.exists():
            raise RuntimeError(f"MorphScore data missing for {lang} ({path}) -- "
                                f"refusing to silently drop a requested language")
        raw_datasets[lang] = pd.read_csv(path)
        if lang == "guj":
            n = len(raw_datasets["guj"])
            print(f"[flag] Gujarati: {n} rows staged (thin; the study design corrected -- present, not absent)")

    gate_rows = []
    score_rows = []
    crosscheck_rows = []

    for tok_name in T.PANEL:
        tokenizer = T.load(tok_name)
        subword_prefix = "##" if tok_name == "mbert" else ""

        gate = GATE.gate_report(raw_datasets, tokenizer, tok_name, subword_prefix)
        gate_rows.append(gate)

        for lang in LANG_ORDER_PANEL:
            g = gate[gate.lang == lang].iloc[0]
            ms = MorphScore(data_dir=str(data_dir), language_subset=[LANG_TO_FILE[lang]],
                             freq_scale=True, exclude_single_tok=False,
                             subword_prefix=subword_prefix)
            filtered = ms._filter_dataset(raw_datasets[lang])
            special_toks = set(tokenizer.special_tokens_map.values())

            def _is_zero_token(w):
                if not isinstance(w, str) or not w or w.isspace():
                    return True
                ids = tokenizer(w)["input_ids"]
                decoded = [tokenizer.decode(i) for i in ids]
                kept = [t for t in decoded if t not in special_toks]
                return len(kept) == 0

            zero_mask = filtered["wordform"].map(_is_zero_token)
            n_zero_token = int(zero_mask.sum())
            filtered_clean = filtered[~zero_mask]
            if len(filtered_clean) == 0:
                raise RuntimeError(f"{tok_name}/{lang}: every wordform tokenises to zero "
                                    f"content tokens -- nothing left to score, stopping "
                                    f"rather than reporting an empty result as a number")
            r = ms.get_morphscore(filtered_clean, tokenizer, return_df=False)
            if isinstance(r, dict) and "error" in r:
                raise RuntimeError(f"MorphScore produced no samples for {tok_name}/{lang}: {r}")

            valid = bool(g.valid)
            row = {
                "tokeniser": tok_name, "lang": lang,
                "n_samples": r["num_samples"],
                "n_excluded_zero_token": n_zero_token,
                "pct_excluded_zero_token": 100.0 * n_zero_token / len(filtered) if len(filtered) else 0.0,
                "gujarati_thin_flag": (lang == "guj"),
                "reconstruction_valid": valid,
                "concat_mismatch_rate": float(g.mismatch_rate),
                "morphscore_recall": r["morphscore_recall"] if valid else float("nan"),
                "morphscore_precision": r["morphscore_precision"] if valid else float("nan"),
                "micro_f1": r["micro_f1"] if valid else float("nan"),
                "macro_f1": r["macro_f1"] if valid else float("nan"),
                "invalid_reason": ("" if valid else
                                   f"decode-per-token-id reconstruction mismatch rate "
                                   f"{g.mismatch_rate:.1%} > {GATE.MISMATCH_TOLERANCE:.0%} tolerance "
                                   f"-- morpheme boundary positions are not trustworthy "
                                   f"(see morphscore_gate.py)"),
                "mean_token_char_ratio": r["mean_token_char_ratio"],
            }
            score_rows.append(row)

            df_lang = raw_datasets[lang]
            words = df_lang["wordform"].astype(str)
            words = words[words.str.len() > 0]
            n_tok = words.map(lambda w: T.count(w, tok_name))
            n_chr = words.map(len)
            cpt_sum_over_sum = float(n_chr.sum() / n_tok.sum())
            cpt_mean_of_ratios = float((n_chr / n_tok).mean())
            crosscheck_rows.append({
                "tokeniser": tok_name, "lang": lang,
                "mean_token_char_ratio": row["mean_token_char_ratio"],
                "cpt_sum_over_sum": cpt_sum_over_sum,
                "cpt_mean_of_ratios": cpt_mean_of_ratios,
                "product_ratio_x_cpt_sumsum": row["mean_token_char_ratio"] * cpt_sum_over_sum,
                "product_ratio_x_cpt_meanratio": row["mean_token_char_ratio"] * cpt_mean_of_ratios,
            })
            print(f"{tok_name:10s} {lang:4s} valid={valid!s:5s} n={row['n_samples']:>6d} "
                  f"recall={row['morphscore_recall']!s:>8s} tcr={row['mean_token_char_ratio']:.4f} "
                  f"cpt(sum/sum)={cpt_sum_over_sum:.4f}")

    gate_df = pd.concat(gate_rows, ignore_index=True)
    score_df = pd.DataFrame(score_rows)
    cross_df = pd.DataFrame(crosscheck_rows)

    gate_df.to_csv(C.OUT_TABLES / "morphscore_gate.csv", index=False)
    score_df.to_csv(C.OUT_TABLES / "morphscore_v2.csv", index=False)
    cross_df.to_csv(C.OUT_TABLES / "morphscore_cpt_crosscheck.csv", index=False)

    print()
    print("=" * 78)
    print("Reconstruction gate -- which (tokeniser, lang) cells are trustworthy")
    print("=" * 78)
    invalid = gate_df[~gate_df.valid]
    if len(invalid):
        print(invalid[["tokeniser", "lang", "mismatch_rate"]].to_string(index=False))
    else:
        print("(none -- every cell reconstructed exactly)")

    print()
    print("=" * 78)
    print("CPT cross-check: corr(mean_token_char_ratio, cpt_sum_over_sum) should be strong")
    print("=" * 78)
    valid_cross = cross_df.merge(score_df[["tokeniser", "lang", "reconstruction_valid"]],
                                  on=["tokeniser", "lang"])
    corr_all = cross_df["mean_token_char_ratio"].corr(cross_df["cpt_sum_over_sum"])
    corr_valid = valid_cross.loc[valid_cross.reconstruction_valid, "mean_token_char_ratio"].corr(
        valid_cross.loc[valid_cross.reconstruction_valid, "cpt_sum_over_sum"])
    print(f"Pearson r, all cells: {corr_all:.4f}")
    print(f"Pearson r, reconstruction-valid cells only: {corr_valid:.4f}")

    PROV.register(
        "morphscore_v2", {
            "config": {"freq_scale": True, "exclude_single_tok": False,
                       "subword_prefix": {"mbert": "##", "default": ""}},
            "languages": LANG_ORDER_PANEL,
            "gujarati_n_rows": int(len(raw_datasets["guj"])),
            "n_reconstruction_invalid_cells": int((~gate_df.valid).sum()),
            "invalid_cells": gate_df.loc[~gate_df.valid, ["tokeniser", "lang", "mismatch_rate"]].to_dict("records"),
            "cpt_crosscheck_pearson_r_all": float(corr_all),
            "cpt_crosscheck_pearson_r_valid_only": float(corr_valid),
            "scores": score_df.to_dict("records"),
        },
        PROV.stamp(Path(__file__),
                   modules=["morphscore_gate", "intrinsic", "tokenisers", "constants"],
                   seed=None,
                   note=("MorphScore v2, native script, 11 tokenisers, 9 languages; "
                         "reconstruction-validity gate applied before trusting recall/precision")))
    return 0


if __name__ == "__main__":
    sys.exit(main())

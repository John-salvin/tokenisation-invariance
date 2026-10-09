#!/usr/bin/env python3
"""Sentence-level de-romanisation with the IndicXlit engine.

Ran on the GPU cluster; see experiments/README.md.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd

ROOT = Path.home() / ("tokinv_work/repo/panel-pipeline")
sys.path.insert(0, str(ROOT / "src"))
import constants as C

from ai4bharat.transliteration import XlitEngine

LANG_MAP = {"guj": "gu", "tam": "ta", "mal": "ml", "mar": "mr", "hin": "hi"}

INTERIM = Path.home() / ("tokinv_work/repo/cluster_copy/data/interim")
DEROM_FINAL = INTERIM / "restoration_sentencelevel_derom.parquet"
RT_FINAL = INTERIM / "restoration_sentencelevel_roundtrip.parquet"


def derom_path(lang: str) -> Path:
    return INTERIM / f"restoration_sentencelevel_derom_{lang}.parquet"


def rt_path(lang: str) -> Path:
    return INTERIM / f"restoration_sentencelevel_roundtrip_{lang}.parquet"


def main():
    INTERIM.mkdir(parents=True, exist_ok=True)
    seg = pd.read_parquet(C.SEGMENTS_PARQUET)
    seg = seg[seg.mqm_valid].copy()

    pending = [lang for lang in LANG_MAP if not (derom_path(lang).exists() and rt_path(lang).exists())]

    if not pending:
        print("All languages already have per-language outputs on disk; skipping straight to concat.")
    else:
        print("Loading en2indic (roman->indic, for de-romanisation) and "
              "indic2en (indic->roman, for the noise-floor forward leg) engines...")
        t0 = time.time()
        en2indic = XlitEngine(lang2use=list(LANG_MAP.values()), beam_width=4,
                              rescore=False, src_script_type="roman")
        indic2en = XlitEngine(lang2use=list(LANG_MAP.values()), beam_width=4,
                              rescore=False, src_script_type="indic")
        print(f"  loaded in {time.time()-t0:.1f}s")

        for lang, xlit_lang in LANG_MAP.items():
            if derom_path(lang).exists() and rt_path(lang).exists():
                print(f"\n=== {lang} ({xlit_lang}): already done, skipping ===")
                continue

            rom = seg[(seg.lang == lang) & (seg.condition == "romanised")].reset_index(drop=True)
            nat = seg[(seg.lang == lang) & (seg.condition == "native")].reset_index(drop=True)
            print(f"\n=== {lang} ({xlit_lang}): {len(rom)} romanised, {len(nat)} native ===")

            t0 = time.time()
            derom_targets = [en2indic.translit_sentence(str(t), xlit_lang) for t in rom["target"]]
            print(f"  de-romanisation: {time.time()-t0:.1f}s for {len(rom)} segments")
            derom_rows = [{"segment_id": sid, "lang": lang, "derom_target": dt}
                          for sid, dt in zip(rom["segment_id"], derom_targets)]

            t0 = time.time()
            synth_rom = [indic2en.translit_sentence(str(t), xlit_lang) for t in nat["target"]]
            print(f"  native->romanised (synthetic): {time.time()-t0:.1f}s")
            t0 = time.time()
            roundtrip_targets = [en2indic.translit_sentence(t, xlit_lang) for t in synth_rom]
            print(f"  synthetic-romanised->de-romanised (roundtrip): {time.time()-t0:.1f}s")
            rt_rows = [{"segment_id": sid, "lang": lang,
                       "synthetic_romanised": sr, "roundtrip_target": rt}
                       for sid, sr, rt in zip(nat["segment_id"], synth_rom, roundtrip_targets)]

            pd.DataFrame(derom_rows).to_parquet(derom_path(lang), index=False)
            pd.DataFrame(rt_rows).to_parquet(rt_path(lang), index=False)
            print(f"  saved {len(derom_rows)} derom rows and {len(rt_rows)} rt rows for {lang} to disk")

    derom_parts = [pd.read_parquet(derom_path(l)) for l in LANG_MAP if derom_path(l).exists()]
    rt_parts = [pd.read_parquet(rt_path(l)) for l in LANG_MAP if rt_path(l).exists()]

    if derom_parts:
        pd.concat(derom_parts, ignore_index=True).to_parquet(DEROM_FINAL, index=False)
    if rt_parts:
        pd.concat(rt_parts, ignore_index=True).to_parquet(RT_FINAL, index=False)

    n_derom = sum(len(p) for p in derom_parts)
    n_rt = sum(len(p) for p in rt_parts)
    n_done = len(derom_parts)
    print(f"\nWrote {n_derom} de-romanised rows and {n_rt} roundtrip rows "
          f"({n_done}/{len(LANG_MAP)} languages complete)")
    return 0 if n_done == len(LANG_MAP) else 1


if __name__ == "__main__":
    sys.exit(main())

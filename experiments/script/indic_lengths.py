#!/usr/bin/env python3
"""Token lengths of every Indic panel row (both conditions) under each
metric's own tokeniser, for the no-truncation rule (a row enters a
metric's comparison only if that metric scores it untruncated): XLM-R counts of source, translation and reference (COMET family),
the MetricX-24 input in mT5 tokens, the BLEURT-20 pair, and each side in
mBERT tokens (BERTScore). Uses the tokenisers exactly as the scoring code
loads them (experiments/script/score_wmt.py, g2_tokenisers).

Input: the text columns of data/indic/segments.parquet. Output:
data/derived/indic_lengths/indic_lengths.parquet (texts dropped).

    python experiments/script/indic_lengths.py IN.parquet OUT.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_wmt as R  # noqa: E402  (puts the original scoring code on the path)

MP, T = R.MP, R.T


def main() -> None:
    d = pd.read_parquet(sys.argv[1])
    x = lambda s: len(T.tokenise(str(s), "xlmr", add_special_tokens=False))
    d["ntok_src"] = [x(s) for s in d.source_en]
    d["ntok_mt"] = [x(s) for s in d.target]
    d["ntok_ref"] = [x(s) for s in d.reference]
    mt5, blt, mb = R.g2_tokenisers()
    d["ntok_metricx"] = [len(mt5(MP._metricx_input_text(s, m, r))["input_ids"])
                         for s, m, r in zip(d.source_en, d.target, d.reference)]
    d["ntok_bleurt"] = [len(blt(r, m)["input_ids"]) for m, r in zip(d.target, d.reference)]
    d["ntok_mbert_mt"] = [len(mb(m)["input_ids"]) for m in d.target]
    d["ntok_mbert_ref"] = [len(mb(r)["input_ids"]) for r in d.reference]
    d.drop(columns=["target", "reference", "source_en"]).to_parquet(sys.argv[2], index=False)
    print("done", len(d))


if __name__ == "__main__":
    main()

# Data

Everything the notebooks need (37 MB). Nothing has to be downloaded.
`experiments/README.md` says which script produced each file.

## `indic/segments.parquet`: IndicMT Eval in two scripts

14,000 rows: 7,000 segments (5 languages x 1,400), each in two conditions.

| Column | Meaning |
|---|---|
| `segment_id`, `lang`, `condition` | e.g. `guj_0665`, `guj`, `native` or `romanised` |
| `source_en`, `target`, `reference` | English source; MT output and reference in the condition's script |
| `system` | one of the 6 MT systems: IndicTrans_Samanantar, NLLB, bing_api, cvit_iiith, google_api, mT5 |
| `mqm_score`, `mqm_valid` | human MQM score (higher is better); `mqm_valid` is False for the 5 segments without a numeric score |
| `comet_22`, `bleurt`, `bertscore`, `bleu`, `chrf`, `ter`, `comet_qe` | metric scores (COMET-22 is stored x 100) |
| `tp_xlmr`, `tok_count_*` | XLM-R token parity and token counts under four tokenisers |

The native text, the MQM scores and the system labels come from IndicMT Eval.
The romanised condition is ours: IndicXlit at sentence level, beam 4
(`experiments/data_prep/romanise_indicmt.ipynb`). So are all metric scores.
Two features of the corpus are kept as they are:

* **37 Tamil native chrF values exceed 100**, which chrF cannot produce. They
  are left out of chrF's agreement in Table 3; the paper reports both values.
* **58 to 90 percent of native targets contain a no-break space (U+00A0)**.
  A tokeniser decodes every word boundary as a plain space, so these could
  never be re-cut into the identical string; the sampler turns U+00A0 into a
  plain space first.

## `wmt/wmt24_mqm_en-de_en-es.parquet`

WMT24 English to German (6,006 rows) and English to Spanish (4,651 rows) with
MQM scores, from mt-metrics-eval: the two Latin-script settings of Table 1.

## `scores/`: metric scores of every re-segmentation (Section 4)

| File | Rows | Score |
|---|---|---|
| `resegmentation/indic_comet22.parquet` | (segment, bucket, seed) on IndicMT Eval | COMET-22 |
| `resegmentation/indic_samples_meta.parquet` | every sampler draw, including buckets a segment cannot reach | none |
| `resegmentation/indic_metricx24_n300.parquet`, `_n900` | the same with mT5's vocabulary, 300 then 900 segments per language | MetricX-24 (an error score: lower is better) |
| `resegmentation/latin_wmt24_comet22.parquet` | WMT24 German and Spanish | COMET-22 |
| `resegmentation/wmt_<pair>_<run>.parquet` | WMT24/25 pairs; `run` is `base` or `cjk` (first runs) or `hp`, `hp_cjk`, `pw` (runs on 1,200 segments); `cjk` runs use six buckets | COMET-22 |
| `resegmentation/wmt_en-it_IT_kiwi22_pw.parquet`, `wmt_en-et_EE_kiwi22_pw.parquet` | the same rows scored without a reference, because WMT25 Italian has none (its reference is the string "NaN"); Estonian is the control | CometKiwi-22 |
| `metaeval_panel.parquet` | (segment, condition) | all eleven metrics |

Bucket labels are fragmentation-ratio intervals, `[1.00,1.25)` to
`[2.75,3.00)`; `canonical` is the tokeniser's own segmentation.

## `derived/`: other outputs of the GPU runs

| Folder | Content | Paper |
|---|---|---|
| `offtarget/` | language label of every rated WMT output and reference, and the systems that wrote the wrong variety | all WMT analyses |
| `equal_count/` | COMET-22 on re-cuts within 5 percent of the tokeniser's own length | Section 4.2 |
| `layerwise/` | retrieval accuracy per encoder layer and seed | Section 4.3 |
| `effect_predictors/` | token counts, word counts and fragmentation headroom per segment | Limitations |
| `tokeniser_panel/` | sampler checks, token parity under five tokenisers, consistency, MorphScore, no-break-space sensitivity | appendices |
| `indic_lengths/` | each IndicMT Eval row's length under each metric's tokeniser (rows a metric would truncate are left out) | Section 5.1 |
| `beyond_indic/` | seven WMT settings, native and romanised, all eleven metrics, token counts, and the lossy and noise conditions | Sections 5.2, 5.3, 6 |
| `serbian/` | WMT25 English-Serbian, COMET-22 on Cyrillic and Latin | Section 5.2 |
| `adequacy/` | LaBSE retrieval per segment; ISO 15919 and noise conditions with their round-trip error and COMET-22 | Section 6 |
| `repair/` | evaluations of the adapters: held-out folds, lambda frontiers, the hyperparameter grid, the nested selections, WMT transfer | Section 8 |

## Where the original data come from

| Source | Download | Alternative |
|---|---|---|
| IndicMT Eval | [github.com/AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval), folder `Dataset/` | Hugging Face Hub: [`ai4bharat/IndicMTEval`](https://huggingface.co/datasets/ai4bharat/IndicMTEval) |
| WMT24 and WMT25 (texts, system outputs, MQM and ESA scores) | [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval): `python -m mt_metrics_eval.mtme --download` | `wget https://storage.googleapis.com/mt-metrics-eval/mt-metrics-eval-v2.tgz` |
| GlotLID v3 (language identification) | Hugging Face Hub: `cis-lmu/glotlid`, file `model_v3.bin` | [github.com/cisnlp/GlotLID](https://github.com/cisnlp/GlotLID) |
| IndicXlit (romanisation) | `pip install ai4bharat-transliteration==0.5.0.3` | [github.com/AI4Bharat/IndicXlit](https://github.com/AI4Bharat/IndicXlit) |

## Licences and credit

* **IndicMT Eval** (Sai et al., ACL 2023) was created by
  [AI4Bharat](https://github.com/AI4Bharat). The dataset is governed by its own
  licence from [AI4Bharat/IndicMT-Eval](https://github.com/AI4Bharat/IndicMT-Eval).
  Its text and MQM scores are included here only so that every number can be
  reproduced. Please cite it when you use `indic/`.
* **WMT24 and WMT25** data (Kocmi et al., 2024, 2025; Freitag et al., 2024)
  are governed by the terms of
  [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval).
* **Our additions** (romanisations, metric scores, re-segmentations and the
  tables built from them) are released under this repository's MIT licence.

`../CITATIONS.bib` has the BibTeX for every dataset, tool, model and method used.

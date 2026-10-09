# Experiments: the code that produced the data

The notebooks rebuild every table of the paper from the files in `data/`. This
folder holds the code that produced those files: re-segmenting the
translations, scoring them with each metric on GPUs, the probes, and training
the adapters. It ran on a SLURM cluster with H100 GPUs. Each `.sbatch` file is
the exact job that was submitted, with its flags.

| Folder | Paper | What it does |
|---|---|---|
| `data_prep/` | Section 3 | romanise IndicMT Eval and WMT, find off-target WMT systems |
| `tokenisation/` | Section 4 | re-segment, score, the probe, tokeniser statistics |
| `script/` | Section 5 | score every metric in both scripts, WMT and Serbian |
| `meaning/` | Section 6 | LaBSE retrieval, ISO 15919 with noise, lossy WMT romanisation |
| `repair/` | Section 8 | train and evaluate the adapters, transfer to WMT |
| `common/` | all | modules the scripts above import (sampler, metric loading, ...) |

## Which script made which data file

| Data file | Made by |
|---|---|
| **Section 3: data** | |
| `data/indic/segments.parquet` | IndicMT Eval workbooks + `data_prep/romanise_indicmt.ipynb` (IndicXlit, sentence level, beam 4) via `common/ingest.py`; metric columns from `script/` |
| `data/derived/offtarget/` | `data_prep/offtarget_lid.py` (GlotLID v3) |
| romanised WMT text (input to `script/score_wmt.py`) | `data_prep/romanise_wmt.py` (ISO 15919, ISO 9, ISO 233-style; round trip checked) |
| **Section 4: Tokenisation Invariance** | |
| `data/scores/resegmentation/indic_samples_meta.parquet` | `tokenisation/run_reseg_sample.py` (uniformity first checked by `run_sampler_verify.py`) |
| `data/scores/resegmentation/indic_comet22.parquet` | `tokenisation/run_reseg_score.py` |
| `data/scores/resegmentation/indic_metricx24_n300.parquet`, `_n900` | `tokenisation/run_reseg_sample_mt5.py`, then `run_reseg_score_mt5.py` |
| `data/scores/resegmentation/latin_wmt24_comet22.parquet` | `tokenisation/latin_wmt24_reseg.py` (`latin_wmt24.sbatch`) |
| `data/scores/resegmentation/wmt_*_base.parquet`, `_cjk` | `tokenisation/wmt_reseg.py` (`wmt_reseg.sbatch`) |
| `data/scores/resegmentation/wmt_*_hp.parquet`, `_hp_cjk` | the same, 1,200 segments (`wmt_reseg_1200.sbatch`) |
| `data/scores/resegmentation/wmt_*_pw.parquet` | the same, 1,200 segments, five WMT25 pairs (`wmt_reseg_1200_wmt25.sbatch`) |
| `data/scores/resegmentation/wmt_*_kiwi22_pw.parquet` | the same with CometKiwi-22, Italian and Estonian (`wmt_reseg_kiwi.sbatch`) |
| `data/derived/equal_count/` | `tokenisation/run_equal_count_sample.py`, then `run_equal_count_score.py` |
| `data/derived/layerwise/` | `tokenisation/run_layerwise.py`, `run_layerwise_bootstrap.py` |
| `data/derived/effect_predictors/` | `tokenisation/effect_covariates.py` |
| `data/derived/tokeniser_panel/` | `tokenisation/run_tokeniser_panel.py`, `run_morphscore.py`, `run_nbsp_sensitivity.py`, `run_sampler_verify*.py`, `wmt_tokeniser_stats.py`; `tcr_consistency.csv` from `common/consistency.py` |
| **Section 5: Script Invariance** | |
| `data/scores/metaeval_panel.parquet` | `script/run_unified_metric_panel.py` (CometKiwi, xCOMET), `run_metricx_verify_and_score.py`, `run_bleurt_verify_and_score.py`, merged by `build_panel.py` |
| `data/derived/indic_lengths/` | `script/indic_lengths.py` |
| `data/derived/beyond_indic/<pair>_<metric>.parquet` | `script/score_wmt.py` (`score_wmt.sbatch`); the `_tokens` and `_tokens_g2` files are its `--metric tokens` and `--metric tokens_g2` runs |
| `data/derived/serbian/` | `script/serbian.py` with `srp_translit.py` (`serbian.sbatch`) |
| **Section 6: meaning** | |
| `data/derived/adequacy/labse_retrieval_segments.parquet` | `meaning/labse_stratify.py` |
| `data/derived/adequacy/iso_dose_segments.parquet` | `meaning/iso_romanise_noise.py`, `iso_score.py`, `deromanise_indicxlit.py`, then `build_adequacy_inputs.py` |
| `data/derived/beyond_indic/<pair>_comet22_dose.parquet`, `_dose_tokens` | `meaning/wmt_dose_conditions.py`, `wmt_dose_tokens.py`, scored by `script/score_wmt.py --dose` |
| **Section 8: repair** | |
| `data/derived/repair/lolo_plain/` | `repair/lora_lolo_pipeline.py` (A1, one held-out language per run) |
| `data/derived/repair/frontier_*.csv` | `repair/anchored_train.py` (A2; `anchored_train_v2.sbatch`), evaluated across lambda by `lambda_frontier.py` (`anchored_frontier.sbatch`) |
| `data/derived/repair/frontier_grid/` | the same over ranks and loss weights (`grid_train.sbatch`, `grid_eval.sbatch`; 12-hour jobs chained by `chain_submit.sh`) |
| `data/derived/repair/nested/` | `repair/nested_lambda.py` (`nested_lambda.sbatch`), selected by `nested_lambda_select.py` |
| `data/derived/repair/mu_nested/` | `mu_nested_train.sbatch`, `mu_nested_eval.sbatch`, selected by `mu_nested_select.py` |
| `data/derived/repair/transfer/` | `tokenisation/wmt_reseg.py --adapter ...` (`transfer_eval.sbatch`, `transfer_lambda_sweep.sbatch`, `transfer_selected_lambda.sbatch`); cross-script adapters from `crossscript_train.py` (`crossscript_train_znorm.sbatch`, data from `lora_data_wmt.py`) |

`anchored_train.sbatch` is the first A2 run and `anchored_long.sbatch` a
six-epoch run; both appear in the transfer comparison. `mu_nested_train.sbatch`
calls `nested_mu.py`, which is `nested_lambda.py` with its output folder taken
from the environment:

    sed 's|^OUT = ROB / "results" / "tables" / "nested"|OUT = Path(os.environ["MU_OUT"])|' \
        repair/nested_lambda.py > nested_mu.py

## Running these scripts

They need a GPU and `pip install -r requirements-gpu.txt`. They ran from a
working copy laid out as `scripts/` and `src/`, under `$TOKINV_WORK_ROOT`
(default `~/tokinv_work`), with models and the WMT data copied to
`$TOKINV_WORK_ROOT/staging/` because the cluster had no internet access. In
this repository the scripts are in the folders above and the `src/` modules
are in `common/`, so add it to the path first:

    export PYTHONPATH=$PWD/experiments/common:$PYTHONPATH

The WMT data come from [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval);
pass its root with `--mtme` or `MTME_ROOT`. The metrics are
`Unbabel/wmt22-comet-da`, `Unbabel/wmt22-cometkiwi-da`,
`Unbabel/wmt23-cometkiwi-da-xl`, `Unbabel/XCOMET-XL`, `Unbabel/XCOMET-XXL`
(some need a licence accepted on the Hugging Face Hub),
`google/metricx-24-hybrid-large-v2p6`, BLEURT-20 and
`sentence-transformers/LaBSE`.

Scores from these scripts match the committed ones to floating-point noise
(within 1e-5 in our checks), not bit for bit, because GPU arithmetic varies
between hardware and library versions. That is far below the precision of
any number in the paper.

The trained adapters are downloaded by `scripts/download_adapters.py`.

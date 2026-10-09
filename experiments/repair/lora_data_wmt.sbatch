#!/bin/bash
#SBATCH --job-name=loradata
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
# Training data for the cross-script repair. Three scripts the current adapter
# never saw: Cyrillic, Arabic and a second Latin language, alongside the five
# Indic languages already in the pipeline.
$PY -u scripts/beyond/lora_data_wmt.py \
    --pairs wmt25:en-ru_RU:rus wmt25:en-ar_EG:ara wmt25:en-cs_CZ:ces \
    --n-seg 400 --out data/interim/lora_train_wmt.parquet
echo "end=$(date -Is)"

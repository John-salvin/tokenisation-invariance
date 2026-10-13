#!/bin/bash
#SBATCH --job-name=latin_wmt24
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -euo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export N_SEG=300 BATCH=48
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
$PY -u scripts/beyond/latin_wmt24_reseg.py
echo "end=$(date -Is)"

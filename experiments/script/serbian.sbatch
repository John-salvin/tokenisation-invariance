#!/bin/bash
#SBATCH --job-name=serbian
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
export BATCH=32
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
$PY -u scripts/beyond/serbian.py
echo "end=$(date -Is)"

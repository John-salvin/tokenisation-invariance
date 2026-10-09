#!/bin/bash
#SBATCH --job-name=anch_frontier
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
export ADAPTER_ROOT=$HOME/tokinv_work/staging/lora_anchored_e3
export ADAPTER_PATTERN='checkpoints/lora_anchored_{held}_final.pt'
export OUT_TAG=_anchored
export PER_BUCKET=1500 BATCH=48
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
$PY -u scripts/lora/lambda_frontier.py
echo "end=$(date -Is)"

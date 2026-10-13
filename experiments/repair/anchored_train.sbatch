#!/bin/bash
#SBATCH --job-name=anchored_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -euo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
cd "$REPO"
echo "host=$(hostname) start=$(date -Is) fold=${FOLD}"
$PY -u scripts/lora/anchored_train.py --held_out ${FOLD} \
    --lambda_sd ${LSD:-0.5} --lambda_anch ${LANCH:-1.0} \
    --out_dir $HOME/tokinv_work/staging/lora_anchored/${FOLD}
echo "end=$(date -Is)"

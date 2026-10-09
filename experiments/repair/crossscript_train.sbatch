#!/bin/bash
#SBATCH --job-name=crossscript_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
for held in rus ara ces; do
  echo "===== held-out script: $held ====="
  $PY -u scripts/lora/crossscript_train.py --held_out $held \
      --out_dir $HOME/tokinv_work/staging/lora_crossscript/$held \
      || echo "FAILED fold $held"
done
echo "end=$(date -Is)"

#!/bin/bash
#SBATCH --job-name=anch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp/%x_%A_%a.out
#SBATCH --error=logs/exp/%x_%A_%a.err
#SBATCH --array=0-4
set -euo pipefail
FOLDS=(guj tam mal mar hin)
FOLD=${FOLDS[$SLURM_ARRAY_TASK_ID]}
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
cd "$REPO"
echo "host=$(hostname) start=$(date -Is) fold=$FOLD epochs=3"
$PY -u scripts/lora/anchored_train.py --held_out $FOLD --epochs 3 \
    --lambda_sd 0.5 --lambda_anch 1.0 \
    --out_dir $HOME/tokinv_work/staging/lora_anchored_e3/$FOLD
echo "end=$(date -Is)"

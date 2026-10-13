#!/bin/bash
#SBATCH --job-name=nested
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/exp/%x_%A_%a.out
#SBATCH --error=logs/exp/%x_%A_%a.err
#SBATCH --array=0-19
set -euo pipefail
# 20 (outer X, inner V) pairs, X != V, in a fixed order so array index -> pair is
# reproducible from the index alone.
PAIRS=(
  guj:tam guj:mal guj:mar guj:hin
  tam:guj tam:mal tam:mar tam:hin
  mal:guj mal:tam mal:mar mal:hin
  mar:guj mar:tam mar:mal mar:hin
  hin:guj hin:tam hin:mal hin:mar
)
PAIR=${PAIRS[$SLURM_ARRAY_TASK_ID]}
OUTER=${PAIR%%:*}
INNER=${PAIR##*:}

export TOKINV_WORK_ROOT=$HOME/tokinv_work
REPO=$TOKINV_WORK_ROOT/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export PER_BUCKET=1500 BATCH=48
cd "$REPO"
echo "host=$(hostname) start=$(date -Is) outer=$OUTER inner=$INNER idx=$SLURM_ARRAY_TASK_ID"
$PY -u scripts/lora/nested_lambda.py --outer "$OUTER" --inner "$INNER" --epochs 3
echo "end=$(date -Is)"

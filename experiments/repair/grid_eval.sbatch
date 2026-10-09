#!/bin/bash
# Evaluate every config of the frontier grid. Chained after the training links
# with --dependency=afterok, and resumable: a config whose output CSV already
# exists is skipped, so a link that runs out of wall time loses nothing.
#SBATCH --job-name=grid_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:55:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
ROOT=$HOME/tokinv_work/staging/lora_frontier
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
cd "$REPO"
SOFT=${SOFT:-39600}
DEADLINE=$(( $(date +%s) + SOFT ))
echo "host=$(hostname) start=$(date -Is)"
for r in 4 8 16 32; do
 for sd in 0.0 0.5 1.0; do
  CFG="r${r}_sd${sd}_an1.0"
  [ -d "$ROOT/$CFG" ] || { echo "[skip] $CFG not trained yet"; continue; }
  # every fold of the config must be trained before it is worth evaluating
  n=$(find "$ROOT/$CFG" -name 'fold_*_DONE' 2>/dev/null | wc -l)
  if [ "$n" -lt 5 ]; then echo "[skip] $CFG only $n/5 folds trained"; continue; fi
  # the analysis CSV is written only after a config finishes all five folds,
  # so its presence is a safe "already done" marker. The name must match what
  # lambda_frontier.py actually writes, or every link re-evaluates
  # everything from scratch.
  if [ -f "$REPO/results/tables/lora_lambda_frontier_analysis_${CFG}.csv" ]; then
    echo "[skip] $CFG already evaluated"; continue
  fi
  LEFT=$(( DEADLINE - $(date +%s) ))
  if [ "$LEFT" -lt 600 ]; then echo "[link] ${LEFT}s left, stopping"; exit 0; fi
  echo "=== evaluate $CFG (r=$r) ==="
  export ADAPTER_ROOT="$ROOT/$CFG"
  export ADAPTER_PATTERN="checkpoints/lora_anchored_{held}_final.pt"
  export OUT_TAG="_${CFG}"
  export LORA_R=$r LORA_ALPHA=$(( 2 * r ))
  $PY -u scripts/lora/lambda_frontier.py || echo "FAILED $CFG"
 done
done
echo "end=$(date -Is)"

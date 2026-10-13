#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:55:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
# Held-out lambda frontier of the outer adapters of the mu grid, exactly as
# scripts/lora/grid_eval.sbatch evaluates the Appendix H grid.
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
ROOT=$HOME/tokinv_work/staging/lora_frontier
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
cd "$REPO"
echo "host=$(hostname) start=$(date -Is) cfgs=$CFGS"
for CFG in $CFGS; do
  n=$(find "$ROOT/$CFG" -name "fold_*_DONE" | wc -l)
  [ "$n" -lt 5 ] && { echo "[skip] $CFG $n/5"; continue; }
  [ -f "$REPO/results/tables/lora_lambda_frontier_analysis_${CFG}.csv" ] && { echo "[skip] $CFG done"; continue; }
  echo "=== evaluate $CFG"
  ADAPTER_ROOT="$ROOT/$CFG" ADAPTER_PATTERN="checkpoints/lora_anchored_{held}_final.pt" OUT_TAG="_${CFG}" \
    LORA_R=8 LORA_ALPHA=16 $PY -u scripts/lora/lambda_frontier.py || echo "FAILED $CFG"
done
echo "end=$(date -Is)"

#!/bin/bash
# One link of the chained hyperparameter frontier.
#
# Grid: LoRA rank r in {4,8,16,32} (alpha held at 2r so alpha/r is constant)
# crossed with lambda_sd in {0.0, 0.5, 1.0}, lambda_anch fixed at 1.0, over all
# five LOLO folds. 12 configs x 5 folds = 60 training runs at roughly 33 min
# each, so about 33 GPU-hours across three or four links.
#
# The whole grid is reported rather than a selected best. Selecting a winner
# would need an inner validation fold to stay honest; reporting the frontier
# answers "is the repair a lucky hyperparameter" directly and cannot leak.
#
# Every (config, fold) has its own out_dir and therefore its own DONE marker, so
# a link resumes the grid exactly where the previous one stopped.
#SBATCH --job-name=grid_train
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
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
mkdir -p "$ROOT"
EPOCHS=${EPOCHS:-3}
SOFT=${SOFT:-39600}
DEADLINE=$(( $(date +%s) + SOFT ))
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) epochs=$EPOCHS soft=${SOFT}s"

TOTAL=0; DONE=0
for r in 4 8 16 32; do
 for sd in 0.0 0.5 1.0; do
  CFG="r${r}_sd${sd}_an1.0"
  for held in guj tam mal mar hin; do
    TOTAL=$(( TOTAL + 1 ))
    OUT="$ROOT/$CFG/$held"
    if [ -f "$OUT/fold_${held}_DONE" ]; then
      DONE=$(( DONE + 1 )); continue
    fi
    LEFT=$(( DEADLINE - $(date +%s) ))
    if [ "$LEFT" -lt 600 ]; then
      echo "[link] ${LEFT}s left, stopping; next link continues at $CFG/$held"
      echo "progress=${DONE}/${TOTAL}_seen end=$(date -Is)"
      exit 0
    fi
    echo "=== $CFG fold $held (budget ${LEFT}s) ==="
    $PY -u scripts/lora/anchored_train.py --held_out "$held" \
        --epochs "$EPOCHS" --r "$r" --alpha "$(( 2 * r ))" \
        --lambda_sd "$sd" --lambda_anch 1.0 \
        --ckpt_every_steps 200 --max_seconds "$LEFT" \
        --out_dir "$OUT" || { echo "FAILED $CFG $held"; exit 1; }
    [ -f "$OUT/fold_${held}_DONE" ] && DONE=$(( DONE + 1 ))
  done
 done
done
echo "grid_complete=${DONE}/${TOTAL} end=$(date -Is)"

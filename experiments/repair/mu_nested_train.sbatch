#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:55:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
# Nested mu selection (see experiments/README.md). Each worker claims
# tasks from tasks.txt with an atomic mkdir, skips finished ones, and stops
# before the wall if less than 50 minutes remain.
set -uo pipefail
M=$HOME/tokinv_work/mu_nested
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
ROOT=$HOME/tokinv_work/staging/lora_frontier
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2 PER_BUCKET=1500 BATCH=48
cd "$REPO"
DEADLINE=$(( $(date +%s) + 42600 ))
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
i=0
while read -r kind sd an a b; do
  i=$((i+1))
  if [ "$kind" = outer ]; then
    CFG="r8_sd${sd}_an${an}"; OUT="$ROOT/$CFG/$a"; DONE="$OUT/fold_${a}_DONE"
  else
    OUTD="$M/nested/sd${sd}_an${an}"; DONE="$OUTD/DONE_${a}__${b}.json"
  fi
  [ -f "$DONE" ] && continue
  mkdir "$M/claims/$i" 2>/dev/null || continue
  LEFT=$(( DEADLINE - $(date +%s) ))
  if [ "$LEFT" -lt 3000 ]; then rmdir "$M/claims/$i"; echo "[stop] ${LEFT}s left"; break; fi
  echo "=== task $i: $kind $sd $an $a $b start=$(date -Is)"
  if [ "$kind" = outer ]; then
    $PY -u scripts/lora/anchored_train.py --held_out "$a" --epochs 3 --r 8 --alpha 16 \
        --lambda_sd "$sd" --lambda_anch "$an" --ckpt_every_steps 200 --max_seconds "$LEFT" \
        --out_dir "$OUT" || echo "FAILED task $i"
  else
    mkdir -p "$OUTD"
    MU_OUT="$OUTD" $PY -u $M/nested_mu.py --outer "$a" --inner "$b" --epochs 3 \
        --lambda_sd "$sd" --lambda_anch "$an" || echo "FAILED task $i"
  fi
  [ -f "$DONE" ] || { echo "NOTDONE task $i"; rmdir "$M/claims/$i"; }
done < "$M/tasks.txt"
echo "end=$(date -Is)"

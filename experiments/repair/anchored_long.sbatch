#!/bin/bash
# One link of a chained long LoRA run.
#
# The gpu partition caps a job at 12h, so a multi-day run has to be split. Each
# link stops itself at SOFT seconds, writing a resume checkpoint, and exits 0 so
# the next link (submitted with --dependency=afterok) starts. A real crash exits
# non-zero and breaks the chain instead of being skipped over. Folds that already
# finished are skipped via their DONE marker, so a link only continues unfinished
# work and the chain is safe to over-provision.
#SBATCH --job-name=anchored_long
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
OUT=$HOME/tokinv_work/staging/lora_long
FOLDS="guj tam mal mar hin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
mkdir -p "$OUT"
EPOCHS=${EPOCHS:-6}
SOFT=${SOFT:-39600}          # 11h of the 11h55m wall, leaving room to save
DEADLINE=$(( $(date +%s) + SOFT ))
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) epochs=$EPOCHS soft=${SOFT}s"

for held in $FOLDS; do
  if [ -f "$OUT/$held/fold_${held}_DONE" ]; then
    echo "[link] $held already complete, skipping"
    continue
  fi
  LEFT=$(( DEADLINE - $(date +%s) ))
  if [ "$LEFT" -lt 600 ]; then
    echo "[link] only ${LEFT}s left, stopping here; the next link continues"
    break
  fi
  echo "=== fold $held (budget ${LEFT}s) ==="
  $PY -u scripts/lora/anchored_train.py --held_out "$held" \
      --epochs "$EPOCHS" --ckpt_every_steps 200 --max_seconds "$LEFT" \
      --out_dir "$OUT/$held" || { echo "FAILED $held"; exit 1; }
done

ALL_DONE=1
for held in $FOLDS; do
  [ -f "$OUT/$held/fold_${held}_DONE" ] || ALL_DONE=0
done
echo "all_folds_complete=$ALL_DONE end=$(date -Is)"

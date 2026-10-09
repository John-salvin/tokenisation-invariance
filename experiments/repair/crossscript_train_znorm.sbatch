#!/bin/bash
#SBATCH --job-name=transfer_crossscript_training
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
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
# Re-run with labels z-normalised within language. The first attempt mixed
# MQM (0-25) and ESA (0-100) into one globally normalised target.
for held in rus ara ces; do
  echo "===== held-out script: $held ====="
  $PY -u scripts/lora/crossscript_train.py --held_out $held \
      --out_dir $HOME/tokinv_work/staging/lora_crossscript_fixed/$held || echo "FAILED $held"
done
echo "--- evaluate ---"
CF=$HOME/tokinv_work/staging/lora_crossscript_fixed
for spec in rus:wmt25:en-ru_RU ara:wmt25:en-ar_EG ces:wmt25:en-cs_CZ; do
  h=${spec%%:*}; rest=${spec#*:}; ts=${rest%%:*}; lp=${rest##*:}
  echo "=== CROSSFIX $h on $lp ==="
  $PY -u scripts/beyond/wmt_reseg.py --testset $ts --pair $lp --n-seg 600 \
      --adapter $CF/$h/checkpoints/lora_anchored_${h}_final.pt \
      --tag "${lp}_crossfix" || echo "FAILED eval $h"
done
echo "end=$(date -Is)"

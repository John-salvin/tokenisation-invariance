#!/bin/bash
#SBATCH --job-name=repair_eval
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
CROSS=$HOME/tokinv_work/staging/lora_crossscript

# (1) Sample-matched baselines at N=600. The earlier base runs used N=300 or
# N=1200, so repair and cost could not be read off them cleanly.
for spec in wmt25:en-ru_RU wmt25:en-ar_EG wmt25:en-cs_CZ wmt25:en-is_IS wmt25:en-uk_UA wmt24:en-hi wmt24:en-cs; do
  ts=${spec%%:*}; lp=${spec##*:}
  echo "=== BASE600 $ts $lp ==="
  $PY -u scripts/beyond/wmt_reseg.py --testset $ts --pair $lp \
      --n-seg 600 --tag "${lp}_base600" || echo "FAILED base600 $lp"
done

# (2) Cross-script adapters: each was trained with its own script held out
# entirely, so this is held-out script rather than held-out language.
echo "=== CROSS rus on en-ru_RU ==="
$PY -u scripts/beyond/wmt_reseg.py --testset wmt25 --pair en-ru_RU --n-seg 600 \
    --adapter $CROSS/rus/checkpoints/lora_anchored_rus_final.pt --tag en-ru_RU_cross || echo "FAILED cross rus"
echo "=== CROSS ara on en-ar_EG ==="
$PY -u scripts/beyond/wmt_reseg.py --testset wmt25 --pair en-ar_EG --n-seg 600 \
    --adapter $CROSS/ara/checkpoints/lora_anchored_ara_final.pt --tag en-ar_EG_cross || echo "FAILED cross ara"
echo "=== CROSS ces on en-cs_CZ ==="
$PY -u scripts/beyond/wmt_reseg.py --testset wmt25 --pair en-cs_CZ --n-seg 600 \
    --adapter $CROSS/ces/checkpoints/lora_anchored_ces_final.pt --tag en-cs_CZ_cross || echo "FAILED cross ces"
echo "end=$(date -Is)"

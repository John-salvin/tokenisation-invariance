#!/bin/bash
#SBATCH --job-name=wmt_reseg_1200_wmt25
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
# Five WMT25 pairs at four times the segment pool (1,200 segments). WMT25
# segments are about twice as long as WMT24's, so the 160-token canonical
# filter leaves few of them at the default pool size. Produces the `_pw` files
# in data/scores/resegmentation/.
for lp in en-uk_UA en-it_IT en-et_EE en-is_IS en-ru_RU; do
  echo "=== wmt25 $lp N_SEG=1200 ==="
  N_SEG=1200 $PY -u scripts/beyond/wmt_reseg.py --testset wmt25 --pair "$lp" \
      --n-seg 1200 --tag "${lp}_pw" || echo "FAILED: $lp"
done
echo "end=$(date -Is)"

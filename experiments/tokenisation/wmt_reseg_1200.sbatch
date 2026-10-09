#!/bin/bash
#SBATCH --job-name=hp
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
# COMET-22 at high power on every pair the first run left with a bootstrap
# interval spanning zero, so the evidence rests on tight intervals
# rather than on an anti-conservative permutation p.
run () { local ts=$1 lp=$2; shift 2
  echo "=== HP $ts $lp $* ==="
  $PY -u scripts/beyond/wmt_reseg.py --testset "$ts" --pair "$lp" \
      --metric comet22 --n-seg 1200 "$@" || echo "FAILED: $ts $lp"; }
for lp in en-hi en-ru en-uk en-cs en-is; do run wmt24 "$lp" --tag "${lp}_hp"; done
for lp in en-cs_CZ en-ar_EG en-bho_IN; do run wmt25 "$lp" --tag "${lp}_hp"; done
for lp in en-zh_CN en-ja_JP en-ko_KR; do
  run wmt25 "$lp" --max-ratio 1.75 --n-buckets 6 --pre-normalise --tag "${lp}_hp_cjk"
done
echo "end=$(date -Is)"

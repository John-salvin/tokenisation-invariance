#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
# Full run: one metric over the pairs in $PAIRS, native and romanised.
# Usage: sbatch --job-name=score_<metric> --export=ALL,METRIC=<m>,PAIRS="<p ...>" score_wmt.sbatch
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export PYTHONPATH=$HOME/tokinv_work/pylibs310
WORK=$HOME/tokinv_work/wmt
cd "$REPO"
echo "host=$(hostname) metric=$METRIC pairs=$PAIRS start=$(date -Is)"
for pair in $PAIRS; do
  SUF=${DOSE:+_dose}; INDIR=${DOSE:+$WORK/input_dose}; INDIR=${INDIR:-$WORK/input}
  if [ -f "$WORK/full/${pair}_${METRIC}${SUF}.parquet" ]; then echo "skip $pair (done)"; continue; fi
  $PY -u $WORK/score_wmt.py --input $INDIR/$pair.parquet --metric $METRIC ${DOSE:+--dose} \
      --out $WORK/full --batch-size 16 || echo "FAILED $pair $METRIC"
done
echo "end=$(date -Is)"

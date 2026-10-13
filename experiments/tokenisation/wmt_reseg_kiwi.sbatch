#!/bin/bash
#SBATCH --job-name=wmt_reseg_kiwi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
# Table 1 Italian: WMT25 en-it_IT has no reference (refA is the string NaN), so
# the COMET-22 row is degenerate. The same re-segmentation run with the
# reference-free CometKiwi-22, and Estonian with the same metric as a control.
# Submit once per pair: sbatch --export=ALL,LP=en-it_IT wmt_reseg_kiwi.sbatch
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
echo "host=$(hostname) pair=$LP start=$(date -Is)"
N_SEG=1200 $PY -u scripts/beyond/wmt_reseg.py --testset wmt25 --pair "$LP" --n-seg 1200 \
    --metric cometkiwi22 --tag "${LP}_kiwi22_pw" || echo "FAILED: $LP"
echo "end=$(date -Is)"

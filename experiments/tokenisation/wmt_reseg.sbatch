#!/bin/bash
#SBATCH --job-name=wmt_reseg
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
export N_SEG=${N_SEG:-300} BATCH=${BATCH:-48}
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"

# Standard eight-bucket ladder. Measured beforehand: these scripts reach ratio
# 3.0 and XLM-R round-trips their text, so results line up with the existing
# Indic and Latin rows.
FULL_WMT25="en-ru_RU en-uk_UA en-ar_EG en-bho_IN en-cs_CZ en-et_EE en-is_IS en-it_IT"
FULL_WMT24="en-hi en-ru en-uk en-cs en-is"
# CJK is a different regime on two counts, both measured: XLM-R leaves the text
# near character level (max ratio ~1.9, not 3.0) and its normaliser rewrites
# fullwidth punctuation, so 0/60 Chinese segments survive the byte-identity
# guard untouched. Run on an even six-bucket ladder over the reachable range,
# after pre-normalising to the tokeniser's own representation, and report
# separately from the rows above.
CJK_WMT25="en-zh_CN en-ja_JP en-ko_KR"

run () { local ts=$1 lp=$2; shift 2
  echo "=== $ts $lp $* ==="
  $PY -u scripts/beyond/wmt_reseg.py --testset "$ts" --pair "$lp" "$@" \
    || echo "FAILED: $ts $lp (continuing)"; }

for lp in $FULL_WMT25; do run wmt25 "$lp"; done
for lp in $FULL_WMT24; do run wmt24 "$lp"; done
for lp in $CJK_WMT25; do
  run wmt25 "$lp" --max-ratio 1.75 --n-buckets 6 --pre-normalise --tag "${lp}_cjk"
done
echo "end=$(date -Is)"

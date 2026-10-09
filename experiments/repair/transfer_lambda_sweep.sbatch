#!/bin/bash
# Re-run the cross-script transfer at the Pareto lambda instead of full strength.
# The frontier analysis puts the Pareto point near 0.5 on all five folds, so the
# lambda=1 transfer numbers in sec:repair overstate the canonical cost.
# Both the 3-epoch adapter (the one the paper's numbers come from) and the new
# 6-epoch adapter are run, so the comparison is like for like.
#SBATCH --job-name=tf_lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp/%x_%j.out
#SBATCH --error=logs/exp/%x_%j.err
set -uo pipefail
REPO=$HOME/tokinv_work/repo/cluster_copy
PY=$HOME/.conda/envs/mt_eval/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_TF=0 USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
export MTME_ROOT=$HOME/tokinv_work/staging/mtme/mt-metrics-eval-v2
cd "$REPO"
A3=$HOME/tokinv_work/staging/lora_anchored_e3/guj/checkpoints/lora_anchored_guj_final.pt
A6=$HOME/tokinv_work/staging/lora_long/guj/checkpoints/lora_anchored_guj_final.pt
echo "host=$(hostname) start=$(date -Is)"
for spec in wmt25:en-ar_EG wmt25:en-ru_RU wmt25:en-cs_CZ wmt25:en-is_IS wmt25:en-uk_UA wmt24:en-cs; do
  ts=${spec%%:*}; lp=${spec##*:}
  for lam in 0.5; do
    echo "=== 3ep lam=$lam $lp ==="
    $PY -u scripts/beyond/wmt_reseg.py --testset $ts --pair $lp --n-seg 600 \
        --adapter $A3 --adapter-lambda $lam --tag "${lp}_a3lam${lam}" \
        || echo "FAILED 3ep $lp"
    echo "=== 6ep lam=$lam $lp ==="
    $PY -u scripts/beyond/wmt_reseg.py --testset $ts --pair $lp --n-seg 600 \
        --adapter $A6 --adapter-lambda $lam --tag "${lp}_a6lam${lam}" \
        || echo "FAILED 6ep $lp"
  done
done
echo "end=$(date -Is)"

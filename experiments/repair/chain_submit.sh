#!/bin/bash
# Chain N copies of an sbatch script so each starts only after the previous one
# exits 0. Used to run past the gpu partition's 12h ceiling: the job script must
# checkpoint and stop itself before the wall, then exit 0.
#
#   bash scripts/lora/chain_submit.sh 6 scripts/lora/anchored_long.sbatch
#
# afterok, not afterany, is deliberate. A link that hits the hard wall is marked
# TIMEOUT and would release an afterany successor, hiding the fact that no
# checkpoint was written. With afterok, the chain only advances on a clean,
# checkpointed stop, and a crash stops it where it broke.
set -euo pipefail
N=${1:?number of links}
SCRIPT=${2:?path to sbatch script}
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 1; }
PREV=""
for i in $(seq 1 "$N"); do
  if [ -z "$PREV" ]; then
    JID=$(sbatch --parsable "$SCRIPT")
  else
    JID=$(sbatch --parsable --dependency=afterok:"$PREV" "$SCRIPT")
  fi
  echo "link $i: job $JID${PREV:+ (after $PREV)}"
  PREV=$JID
done
echo "chain of $N links submitted; final job $PREV"

#!/bin/bash
# Train one model on all GPUs of this machine.
#   bash scripts/linux/train.sh configs/<config>.yaml [key=value overrides...]
# Checkpoints go to <out_dir>/last.pt every few hundred steps. If the run is interrupted
# (Ctrl-C, reboot, crash), run the SAME command again and it resumes from last.pt.
# The run counts as finished only when <out_dir>/DONE exists.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
CONFIG=${1:?usage: train.sh configs/<config>.yaml [key=value ...]}; shift
OUT=$(python -m surgseg.train --config "$CONFIG" --override "$@" --print-out-dir)
NGPU=$(num_gpus)
echo "config=$CONFIG out=$OUT gpus=$NGPU"
if [ -f "$OUT/DONE" ]; then echo "already finished"; exit 0; fi
if [ "$NGPU" -le 1 ]; then
  python -m surgseg.train --config "$CONFIG" --override "$@"
else
  torchrun --standalone --nproc_per_node="$NGPU" -m surgseg.train --config "$CONFIG" --override "$@"
fi
[ -f "$OUT/DONE" ] || { echo "ERROR: training stopped before finishing - rerun this command to resume"; exit 1; }

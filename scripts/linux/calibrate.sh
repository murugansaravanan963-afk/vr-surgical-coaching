#!/bin/bash
# Evaluate + fit the calibration temperature for each checkpoint given.
#   bash scripts/linux/calibrate.sh <ckpt.pt> [<ckpt.pt> ...]
# Skips checkpoints that already have <ckpt>.eval_test.json.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
for c in "$@"; do
  if [ -f "${c%.pt}.eval_test.json" ]; then echo "already calibrated: $c"; continue; fi
  python -m surgseg.evaluate --ckpt "$c" --split test --fit-temperature
done

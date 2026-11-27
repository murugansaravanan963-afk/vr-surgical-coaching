#!/bin/bash
# No-training study for the weak classes, on an existing checkpoint (default: student_kd):
#   1. test IoU at long side 640 (what the VR build runs: 512x640), 800 and 960
#   2. per-class logit offsets fitted on VAL at 640, checked on test
#   bash scripts/linux/thin_study.sh [path/to/best.pt]          (1 GPU, ~1 h)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
CK=${1:-$SURGSEG_RUNS/student_kd/best.pt}
for LS in 640 800 960; do
  python -m surgseg.evaluate --ckpt "$CK" --split test --long-side $LS --tag ls$LS
done
python -m surgseg.evaluate --ckpt "$CK" --split test --long-side 640 --fit-class-bias --tag ls640_bias
B="${CK%.pt}"
python scripts/linux/compare_evals.py "$B.eval_test_ls640.json" "$B.eval_test_ls800.json" \
    "$B.eval_test_ls960.json" "$B.eval_test_ls640_bias.json" | tee "$B.thin_study.txt"

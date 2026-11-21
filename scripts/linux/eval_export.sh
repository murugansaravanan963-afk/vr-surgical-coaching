#!/bin/bash
# Evaluate on the test split, fit the calibration temperature, export ONNX and benchmark it.
#   bash scripts/linux/eval_export.sh <run dir> [height width]   (default 512 640, multiples of 32)
# The deliverables (best.pt, results JSON, surgseg.onnx + surgseg.json) are copied to
# $SURGSEG_MODELS/<run name>/.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
RUN=${1:?usage: eval_export.sh <run dir> [height width]}
H=${2:-512}; W=${3:-640}
python -m surgseg.evaluate --ckpt "$RUN/best.pt" --split test --fit-temperature
python -m surgseg.export_onnx --ckpt "$RUN/best.pt" --out "$RUN/deploy/surgseg" --height $H --width $W
python -m surgseg.tools.benchmark --model "$RUN/deploy/surgseg.onnx" --sizes ${H}x${W}
DEST="$SURGSEG_MODELS/$(basename "$RUN")"
mkdir -p "$DEST"
cp "$RUN"/best.pt "$RUN"/*.json "$RUN"/deploy/* "$DEST/"
echo "model: $DEST/surgseg.onnx"

#!/bin/bash
# Fine-tune the student on domain-randomised simulator frames, mixed with the real datasets,
# then evaluate, calibrate and export ONNX. One command:
#
#   bash scripts/linux/finetune_sim.sh sim_finetune_upload.zip
#
# The zip (or folder) holds SurgSegCapture episodes (<episode>/images/*.png + masks/*.png,
# unified ids). Needs the base student at $SURGSEG_RUNS/student_kd/best.pt.
# Result: $SURGSEG_MODELS/student_sim/surgseg.onnx + surgseg.json
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
SRC=${1:?usage: finetune_sim.sh <zip or folder of episodes>}
RAW=$SURGSEG_DATA/raw/sim
rm -rf "$RAW" && mkdir -p "$RAW"
if [[ "$SRC" == *.zip ]]; then
  # zips made on Windows store "a\b\c.png"; unzip warns and stops, so extract in Python
  python - "$SRC" "$RAW" <<'EOF'
import os, sys, zipfile
z, out = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
n = 0
for name in z.namelist():
    path = name.replace("\\", "/")
    if path.endswith("/"):
        continue
    dst = os.path.join(out, path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(z.read(name))
    n += 1
print(f"extracted {n} files")
EOF
else
  cp -r "$SRC"/. "$RAW"/
fi
[ -d "$RAW/sim_finetune_upload" ] && RAW="$RAW/sim_finetune_upload"
echo "episodes: $(ls "$RAW" | tr '\n' ' ')"

python -m surgseg.datasets.prepare_custom --src "$RAW" --label-map integration/unity/sim_labels.yaml \
    --out "$SURGSEG_DATA/processed" --name sim --val-episodes randomized_val

bash scripts/linux/train.sh configs/finetune_sim.yaml
bash scripts/linux/eval_export.sh "$SURGSEG_RUNS/student_sim" 512 640
echo "result: $SURGSEG_MODELS/student_sim/surgseg.onnx"

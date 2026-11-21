#!/bin/bash
# Runs the whole training pipeline on this machine, one stage after another
# (run after download_data.sh and prepare_data.sh):
#   1. NUM_TEACHERS round-1 LemonFM teachers (default 2)
#   2. label stage: calibrate teachers, ensemble pseudo-labels + label completion, merge
#   3. round-2 LemonFM teacher on the completed + pseudo-labelled data
#   4. real-time SurgeNetXL student, distilled from the round-2 teacher
#   5. calibrate, export ONNX, benchmark  -> $SURGSEG_MODELS/
#
#   bash scripts/linux/run_pipeline.sh 2>&1 | tee logs/pipeline.log
# This takes days on a single GPU, so run it inside tmux/screen. Every stage is resumable:
# if anything stops, run the same command again and finished stages are skipped.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
S=scripts/linux
R=$SURGSEG_RUNS
N=${NUM_TEACHERS:-2}

CKPTS=()
for ((i = 0; i < N; i++)); do
  echo "=== teacher $i ==="
  bash $S/train.sh configs/train_teacher_lemonfm.yaml seed=$i "out_dir=$R/teacher_lemonfm_s$i"
  CKPTS+=("$R/teacher_lemonfm_s$i/best.pt")
done
echo "=== label stage ===";      bash $S/label_stage.sh "${CKPTS[@]}"
echo "=== round-2 teacher ===";  bash $S/train.sh configs/train_teacher_lemonfm_r2.yaml
echo "=== calibrate r2 ===";     bash $S/calibrate.sh "$R/teacher_lemonfm_r2/best.pt"
echo "=== student ===";          bash $S/train.sh configs/train_student_surgenetxl.yaml
echo "=== export ===";           bash $S/eval_export.sh "$R/student_surgenetxl"
echo "Pipeline finished. Model: $SURGSEG_MODELS/student_surgenetxl/surgseg.onnx"

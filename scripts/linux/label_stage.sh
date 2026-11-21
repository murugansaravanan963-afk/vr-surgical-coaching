#!/bin/bash
# Calibrate the round-1 teachers, then pseudo-label / complete all training data with
# their ensemble (one shard per GPU), then merge into <dataset>_pl datasets.
#   bash scripts/linux/label_stage.sh teacher1/best.pt [teacher2/best.pt ...]
# Resumable: if interrupted, run the same command again; labelled frames are skipped.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
DONE="$SURGSEG_DATA/processed/PSEUDO_LABELS_DONE"
if [ -f "$DONE" ]; then echo "already finished"; exit 0; fi

bash scripts/linux/calibrate.sh "$@"          # skips teachers that are already calibrated

DATASETS=$(bash scripts/linux/pl_datasets.sh)
NGPU=$(num_gpus); [ "$NGPU" -ge 1 ] || NGPU=1
PIDS=()
trap 'kill -TERM "${PIDS[@]}" 2>/dev/null' INT TERM
for ((g = 0; g < NGPU; g++)); do
  CUDA_VISIBLE_DEVICES=$g python -m surgseg.pseudo_label --ckpts "$@" \
      --data-root "$SURGSEG_DATA/processed" --datasets $DATASETS \
      --shard $g --num-shards "$NGPU" > "logs/pl_shard$g.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
[ $FAIL -eq 0 ] || { echo "ERROR: a labelling shard failed (see logs/pl_shard*.log); rerun to resume"; exit 1; }

bash scripts/linux/merge_pl.sh
touch "$DONE"

#!/bin/bash
# Downloads everything that can be scripted (~45 GB into $SURGSEG_DATA/raw). Run:
#   bash scripts/linux/download_data.sh   (takes a few hours; run it inside tmux/screen)
# Safe to re-run: finished parts are skipped. EndoVis 17/18 need a manual step (see end).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
RAW=$SURGSEG_DATA/raw
mkdir -p "$RAW"

# --- surgical foundation-model backbones (LemonFM 1.8 GB, SurgeNetXL 93 MB) ---
python -m surgseg.backbones

# --- original datasets ---
python -c "from datasets import load_dataset as L; [L('nct-tso/dsad', c) for c in ('single_organ','multi_organ')]"
python -c "import dataset_tools as d; d.download(dataset='SISVSE', dst_dir='$RAW/sisvse_dl')"   || echo "WARNING: SISVSE download failed - continuing"
if [ -f ~/.kaggle/kaggle.json ]; then   # Kaggle > Settings > API > Create New Token
  [ -d "$RAW/cholecseg8k" ] || kaggle datasets download -d newslab/cholecseg8k -p "$RAW/cholecseg8k" --unzip     || echo "WARNING: CholecSeg8k download failed - continuing"
else
  echo "WARNING: no ~/.kaggle/kaggle.json - skipping CholecSeg8k"
fi

# --- extra datasets ---
# Endoscapes2023 (6.3 GB, public, CC BY-NC-SA): cystic artery/duct + 58k unlabeled frames
if [ ! -d "$RAW/endoscapes" ]; then
  wget -q --show-progress -O "$RAW/endoscapes.zip" \
    https://s3.unistra.fr/camma_public/datasets/endoscapes/endoscapes.zip
  unzip -q "$RAW/endoscapes.zip" -d "$RAW" && rm "$RAW/endoscapes.zip"
fi

# SAR-RARP50 (34 GB, CC BY-NC-SA): da Vinci instruments/needle/thread; one zip per video
mkdir -p "$RAW/sarrarp50"
python scripts/linux/list_figshare.py 24932529 24932499 | while read -r NAME URL; do
  D="$RAW/sarrarp50/${NAME%.zip}"
  [ -d "$D" ] && continue
  wget -q -O "$RAW/sarrarp50/$NAME" "$URL"
  unzip -q "$RAW/sarrarp50/$NAME" -d "$D" && rm "$RAW/sarrarp50/$NAME"
  echo "  sarrarp50: $NAME"
done

# m2caiSeg (Hugging Face parquet, small)
python -c "from datasets import load_dataset as L; L('MedOtter/m2caiSeg')"

# --- manual / by request (optional, see README "Extra data") ---
#   EndoVis 2017/2018 : grand-challenge.org registration -> copy to $RAW/endovis17, $RAW/endovis18
#   AutoLaparo        : request form at autolaparo.github.io (uterus + tools)
#   Cholec80/CholecT50, HeiChole, SurgeNetYoutube (HF, accept terms): unlabeled video for
#                       distillation -> python -m surgseg.datasets.prepare_unlabeled
echo "Done. Place EndoVis17/18 under $RAW/endovis17 and $RAW/endovis18 if you have them."

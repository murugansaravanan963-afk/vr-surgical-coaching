#!/bin/bash
# One-time setup on a Linux machine with an NVIDIA GPU (driver for CUDA 12.x) and Python 3.10+.
#   bash scripts/linux/setup_env.sh
# Creates a virtual environment in .venv (override with SURGSEG_ENV) and installs everything.
set -eo pipefail
SURGSEG_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SURGSEG_ENV=${SURGSEG_ENV:-$SURGSEG_HOME/.venv}
PYTHON=${PYTHON:-python3}
"$PYTHON" -m venv "$SURGSEG_ENV"
source "$SURGSEG_ENV/bin/activate"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r "$SURGSEG_HOME/requirements.txt"
pip install kaggle dataset-tools
python - <<'PY'
import torch, segmentation_models_pytorch as smp
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
m = smp.create_model("segformer", "mit_b2", encoder_weights="imagenet", classes=21)
print("smp OK, segformer-b2 params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
PY
echo "Done. In every new shell: source scripts/linux/env.sh"

# Source this in every shell before running the other scripts:
#   source scripts/linux/env.sh
# All paths default to folders inside the repo; override any of them by exporting it first,
# e.g. export SURGSEG_DATA=/mnt/bigdisk/surgseg_data  (the datasets need ~45 GB raw + processed).
SURGSEG_HOME=${SURGSEG_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export SURGSEG_HOME
export SURGSEG_ENV=${SURGSEG_ENV:-$SURGSEG_HOME/.venv}          # Python virtual environment
export SURGSEG_DATA=${SURGSEG_DATA:-$SURGSEG_HOME/data}         # raw + processed datasets
export SURGSEG_RUNS=${SURGSEG_RUNS:-$SURGSEG_HOME/runs}         # training runs / checkpoints
export SURGSEG_MODELS=${SURGSEG_MODELS:-$SURGSEG_HOME/models}   # final exported models
export SURGSEG_WEIGHTS=${SURGSEG_WEIGHTS:-$SURGSEG_HOME/weights} # backbone checkpoints
export PYTHONPATH=$SURGSEG_HOME:$PYTHONPATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
if [ -f "$SURGSEG_ENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$SURGSEG_ENV/bin/activate"
else
  echo "WARNING: no virtual environment at $SURGSEG_ENV - run scripts/linux/setup_env.sh first" >&2
fi
mkdir -p "$SURGSEG_HOME/logs"

# Number of GPUs to use: CUDA_VISIBLE_DEVICES if set, otherwise every GPU nvidia-smi sees.
num_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .
  else
    nvidia-smi -L 2>/dev/null | wc -l
  fi
}

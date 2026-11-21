#!/bin/bash
# Combine the pseudo-label shards into <dataset>_pl datasets.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
python -m surgseg.pseudo_label --merge --data-root "$SURGSEG_DATA/processed" \
    --datasets $(bash scripts/linux/pl_datasets.sh)
python -m surgseg.tools.check_data --root "$SURGSEG_DATA/processed"

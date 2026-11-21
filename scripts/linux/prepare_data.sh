#!/bin/bash
# Converts every raw dataset that is present into the unified format (in parallel,
# one process per dataset; ~16 h on 32 CPU cores with all datasets).
#   bash scripts/linux/prepare_data.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$SURGSEG_HOME"
RAW=$SURGSEG_DATA/raw; OUT=$SURGSEG_DATA/processed
declare -A PIDS

run() { local name=$1; shift; python -m "surgseg.datasets.prepare_$name" "$@" --out "$OUT" \
          > "logs/prep_$name.log" 2>&1 & PIDS[$name]=$!; }

run dsad --src nct-tso/dsad
run m2caiseg --src MedOtter/m2caiSeg
META=$(find "$RAW/sisvse_dl" -maxdepth 3 -name meta.json 2>/dev/null | head -1)
if [ -n "$META" ]; then run sisvse --src "$(dirname "$META")" --fold 1; fi
if [ -d "$RAW/cholecseg8k" ]; then run cholecseg8k --src "$RAW/cholecseg8k"; fi
if [ -d "$RAW/endovis18" ]; then run endovis18 --src "$RAW/endovis18"; fi
if [ -d "$RAW/endovis17" ]; then run endovis17 --src "$RAW/endovis17"; fi
if [ -d "$RAW/endoscapes" ]; then run endoscapes --src "$RAW/endoscapes" --unlabeled-stride 5; fi
if [ -d "$RAW/sarrarp50" ]; then run sarrarp50 --src "$RAW/sarrarp50" --unlabeled-every 120; fi

FAIL=0
for name in "${!PIDS[@]}"; do
  if wait "${PIDS[$name]}"; then echo "OK   $name"; else echo "FAIL $name (see logs/prep_$name.log)"; FAIL=1; fi
done
python -m surgseg.tools.check_data --root "$OUT"
exit $FAIL

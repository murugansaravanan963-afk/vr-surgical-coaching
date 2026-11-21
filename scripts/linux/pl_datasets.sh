#!/bin/bash
# Prints the datasets to pseudo-label: those that were prepared (m2caiSeg is already fully
# labeled for its classes and too small to matter, so it is used as-is).
for d in dsad sisvse endovis18 cholecseg8k endoscapes sarrarp50 endovis17 endoscapes_unl sarrarp50_unl; do
  if [ -f "$SURGSEG_DATA/processed/$d/index.jsonl" ]; then echo -n "$d "; fi
done
echo

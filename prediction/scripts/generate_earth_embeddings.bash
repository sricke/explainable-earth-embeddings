#!/bin/bash
set -e

ROOT=/data/locbench
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/prediction"
export PYTHONPATH="$REPO_ROOT"

DATASETS=(
#   fmow
#  nabirds
#  birdsnap
#  yfcc
  mosaiks.elevation
  mosaiks.forest
  mosaiks.nightlights
  mosaiks.population
  sustainbench
)

for ENCODER in geoclip; do
  for DS in "${DATASETS[@]}"; do
    echo "Generating $ENCODER embeddings for $DS..."
    python generate_embeddings.py \
      --encoder "$ENCODER" \
      --dataset "$DS" \
      --out_dir "$ROOT/${DS//.//}"
  done
done

#!/bin/bash
set -e

ROOT=/data/locbench
MODEL_NAME=${1}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/prediction"
export PYTHONPATH="$REPO_ROOT"

DATASETS=(
#   fmow
#   nabirds
#   birdsnap
#   yfcc
  mosaiks.elevation
  mosaiks.forest
  mosaiks.nightlights
  mosaiks.population
  sustainbench
)

for DS in "${DATASETS[@]}"; do
  python generate_embeddings.py \
    --sparse_model "$MODEL_NAME" \
    --dataset "$DS" \
    --out_dir "$ROOT/${DS//.//}"
done

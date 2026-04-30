#!/bin/bash
set -e

ROOT=/data/locbench
MODEL_PATH=$1  # path to SpLiCE checkpoint
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
    --encoder sparse \
    --model_path "$MODEL_PATH" \
    --dataset "$DS" \
    --dataset_root "$ROOT" \
    --out_dir "$ROOT/${DS//.//}"
done

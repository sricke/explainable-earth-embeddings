#!/usr/bin/env bash
set -euo pipefail

MODEL_BASE="$HOME/outputs/explainable-earth-embeddings/git-10M"

MODELS=(
  "$HOME/outputs/explainable-earth-embeddings/geoclip_init.pt"
  "$MODEL_BASE/txt-open_clip_vit_l__loc-satclip__tproj-linear__lproj-none__tft-lora__lft-only_proj__lora_r-8__loss-clip_symmetric__lr-0.0001__h-73876bb2/best.pt"
  "$MODEL_BASE/txt-open_clip_vit_l__loc-climplicit__tproj-linear__lproj-none__tft-lora__lft-only_proj__lora_r-8__loss-clip_symmetric__lr-0.0001__h-a341ad85/best.pt"
  "$MODEL_BASE/txt-open_clip_vit_l__loc-csp_fmow__tproj-linear__lproj-none__tft-lora__lft-only_proj__lora_r-8__loss-clip_symmetric__lr-0.0001__h-86357a7e/best.pt"
)

CONCEPTS_DIR="data/git-10m/concept_dictionaries"

CONCEPTS=(
  "augmented_concepts.json"
  # "concept_dictionary_n500000.json"
  # "concept_dictionary.json"
  # "test_concepts_one_word.json"
  # "test_concepts_LLM_augmented.json"
)

LAT_LON_CSV="$HOME/data/dense_grid/dense_grid.csv"

for model_path in "${MODELS[@]}"; do
  for concept in "${CONCEPTS[@]}"; do

    echo "Running model: $model_path"
    echo "Using concepts: $concept"

    python splice.py \
      --model_path "$model_path" \
      --lat_lon_csv "$LAT_LON_CSV" \
      --concepts_json "$CONCEPTS_DIR/$concept" 
  done
done
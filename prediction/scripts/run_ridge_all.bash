#!/bin/bash
set -e

# ── config ───────────────────────────────────────────────────────────────────
EMB_ROOT=/data/locbench
EARTH_PREFIX=satclip
SPARSE_MODEL=geoclip_geoyfcc
TOPK=10
RESULTS_DIR=/data/locbench/ridge_results
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/prediction"
export PYTHONPATH="$REPO_ROOT"

DATASETS=(
  mosaiks.elevation
  mosaiks.forest
  mosaiks.nightlights
  mosaiks.population
  sustainbench
  # nabirds
  # birdsnap
  # fmow
  # yfcc
)

mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.tsv"
echo -e "dataset\tencoder\tr2\ttop_concepts" > "$SUMMARY"

for DS in "${DATASETS[@]}"; do
  EMB_DIR="$EMB_ROOT/${DS//.//}"
  OUT="$RESULTS_DIR/${DS//./_}"
  mkdir -p "$OUT"

  for PREFIX in "$EARTH_PREFIX" "$SPARSE_MODEL"; do
    LOG="$OUT/${PREFIX}.txt"
    ARGS="--emb_dir $EMB_DIR --prefix $PREFIX --mode ridge --CV --topk $TOPK"
    [[ "$PREFIX" == "$SPARSE_MODEL" ]] && ARGS="$ARGS --model $SPARSE_MODEL"

    echo "=== $DS / $PREFIX ===" | tee "$LOG"
    python main.py $ARGS 2>&1 | tee -a "$LOG"

    R2=$(grep -oP "test metric: \K[\d\.\-]+" "$LOG" || echo "NA")
    CONCEPTS=$(grep "top concepts:" "$LOG" | sed 's/top concepts: //' || echo "")
    echo -e "$DS\t$PREFIX\t$R2\t$CONCEPTS" >> "$SUMMARY"
  done
done

echo ""
echo "Results saved to $RESULTS_DIR"
echo "Summary: $SUMMARY"

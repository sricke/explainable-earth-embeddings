#!/bin/bash

DEST=~/data/skyscript
mkdir -p "$DEST"

BASE_URL=https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript

# Download metadata
curl -o "$DEST/meta2.zip" "$BASE_URL/meta2.zip"

# Download train split
curl -o "$DEST/SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv" \
    "$BASE_URL/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv"

# Download val split
curl -o "$DEST/SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv" \
    "$BASE_URL/dataframe/SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv"

# Download test split
curl -o "$DEST/SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv" \
    "$BASE_URL/dataframe/SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv"

# Download new file
curl -o "$DEST/SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv" \
    "$BASE_URL/SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv"

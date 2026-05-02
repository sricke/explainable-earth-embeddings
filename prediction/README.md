# Prediction

Predict geospatial labels from location embeddings or sparse codes.

## Pipeline

### 1. Download data
```bash
python prediction/download_locbench.py /path/to/data/dir
```
Each dataset needs split files (`train.csv`/`val.csv`/`test.csv` or `train.json`/`val.json`/`test.json`, or a single `data.csv`/`.json` that gets auto-split) with `lat`, `lon`, and `value` columns.

### 2. Generate embeddings or sparse codes
```bash
# Earth embeddings (satclip or geoclip)
python -m prediction.generate_embeddings \
    --encoder satclip --dataset_root /data/mosaiks/elevation --out_dir /data/embs

# Sparse codes (SpLiCE)
python -m prediction.generate_embeddings \
    --encoder sparse --model_path splice.pt --concepts_json concepts.json \
    --dataset_root /data/mosaiks/elevation --out_dir /data/sparse
```
Saves `{encoder}_{split}.pt` or `{method}_sparse_codes_{split}.pt` per split.

### 3. Train and evaluate
```bash
python -m prediction.main \
    --embeddings_path /data/embs --encoder satclip \
    --model_type ridge --out_dir /data/model
```
Prints test R². With sparse codes + ridge, also prints the top-k most predictive concepts.
Model saved as `model.pkl` (ridge) or `model.pt` (MLP) in `--out_dir`.

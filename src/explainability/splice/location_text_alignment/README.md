# Location-Text Alignment

Fine-tunes a text encoder against a pretrained location encoder using CLIP loss.

## Git-10M Dataset
1. Download with 
```bash
hf download lcybuaa/Git-10M --repo-type dataset --local-dir /path/to/git-10m
```

## Precompute Location Embeddings

Before training, precompute location embeddings for each split to avoid recomputing them every epoch:

```bash
python data/precompute_git10m_embeddings.py \
    --git10m_dir /path/to/git-10m \
    --location_model satclip
```

Output parquets (`train.parquet`, `val.parquet`, `test.parquet`) are written to `<git10m_dir>/<location_model>/` by default (Use `--out_dir` to override)

Available location models: `satclip`, `geoclip`, `climplicit`, `csp_fmow`, `sinr`.

## Create dense grid for SpLiCE evaluation

Generates 100,000 points sampled uniformly at random across land surfaces (using sine-latitude sampling to avoid polar clustering). Used as a fixed evaluation set for SpLiCE decompositions.

**Prerequisites:** a land polygon shapefile, e.g. [Natural Earth 10m land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/).

**Configure paths:** create `src/explainability/splice/paths.py` (the script imports from two directories up):

```python
from pathlib import Path

DATA_ROOT = Path("/path/to/data")        # output root; grid saved to DATA_ROOT/dense_grid/dense_grid.csv
SHAPEFILE = Path("/path/to/ne_10m_land.shp")  # land polygon shapefile
```

**Run:**

```bash
python data/create_dense_grid.py
```

Output: `<DATA_ROOT>/dense_grid/dense_grid.csv` — a CSV with `lat` and `lon` columns, 100,000 rows.


## Setup

1. Update `dataset.path`, `dataset.name`, and `model_save_dir` in [`configs/location_text_alignment.yaml`](../../../../../configs/location_text_alignment.yaml).
2. Run training, specifying the location encoder:

```bash
python main.py --location_encoder satclip
```

Available location encoders: `satclip`, `geoclip`, `climplicit`, `csp_fmow`, `sinr`.

Any config value can be replaced in the `location_text_alignment.yaml` config or overridden from the command line (e.g. `--lr 5e-5`).

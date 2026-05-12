# SPLICE

`__init__.py`, `admm.py`, `splice.py`, and `model.py` are taken directly from [ai4life-group/splice](https://github.com/ai4life-group/splice).

## Steps

### Step 1: Train location-text alignment

See [`location_text_alignment/README.md`](location_text_alignment/README.md) for how to download Git-10M, precompute location embeddings, and train the alignment model. The output is a checkpoint directory containing `best.pt`.

### Step 2: Create dense grid

Generates points sampled uniformly at random across land masses. Used to compute the embedding mean for SPLICE decomposition.

**Prerequisites:** a land polygon shapefile, e.g. [Natural Earth 10m land](https://www.naturalearthdata.com/downloads/10m-physical-vectors/).

**Run:**

```bash
python location_text_alignment/data/create_dense_grid.py \
    --shapefile /path/to/ne_10m_land.shp \
    --out /path/to/dense_grid/dense_grid.csv
```

Output: a CSV with `lat` and `lon` columns at the path given by `--out`.

### Step 3: Run SPLICE demo

Open [`notebooks/splice_demo.ipynb`](../../../notebooks/splice_demo.ipynb) and set:

- `MODEL_DIR`: path to the alignment model checkpoint directory from Step 1
- `DENSE_GRID_CSV`: path to the dense grid CSV from Step 2

The notebook encodes the grid points with the location encoder, computes their mean embedding (used to center embeddings before decomposition), then decomposes arbitrary locations into sparse combinations of text concepts. Users can change which locations they want to decompose. This notebook also creates the SpLiCE decomposition visualizations seen in the paper.

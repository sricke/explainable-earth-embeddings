# CLIP-Surgery inference (GeoCLIP / SatCLIP)

Produces spatial similarity maps that highlight which image regions are most consistent with a given GPS location. The method adapts CLIP Surgery — replacing standard self-attention with V-V ("consistent") attention in the last 6 encoder layers — for both GeoCLIP (HuggingFace Transformers ViT-L/14) and SatCLIP (timm ViT-S/16 on Sentinel-2). Unlike CAM-style post-hoc methods, the modification is applied at inference time with no retraining, and the CLS token used for location prediction is preserved exactly, so similarity scores are unchanged.

## Directory layout

```
src/explainability/clip_surgery/
├── _path_setup.py            # adds repo root and src/ to sys.path
├── inference_utils.py        # shared helpers (data loading, preprocessing, plotting)
├── inference_geoclip.py      # batch CLI runner — GeoCLIP
├── inference_satclip.py      # batch CLI runner — SatCLIP
│
├── geoclip_surgery/          # surgery classes for HuggingFace CLIP (GeoCLIP)
│   ├── load.py               # get_geoclip() factory
│   ├── modeling_clip_surgery.py
│   └── layer_utils.py
│
├── satclip_surgery/          # surgery classes for timm ViT (SatCLIP)
│   ├── load.py               # get_satclip() factory
│   ├── model_surgery.py
│   ├── main_surgery.py
│   ├── modified_attention.py
│   └── surgery_vision_transformer.py
│
├── CLIP_Surgery/             # vendored CLIP-Surgery CLIP wrapper
│   └── clip/
│
└── out/                      # inference output (auto-created)
    ├── geoclip/
    └── satclip/
```

---

## Quick start — demo notebook

The interactive demo is at `notebooks/clip_surgery_demo.ipynb`.  

Place your demo images in `notebooks/`:

| File | Description |
|---|---|
| `city.png` | Any RGB photo (GeoCLIP input) |
| `satellite.tif` | Sentinel-2 GeoTIFF, 12 or 13 bands (SatCLIP input) |
| `satellite.png` | Optional RGB preview of the satellite tile for display |

Edit `QUERY_LON` / `QUERY_LAT` (and the `_SAT` variants) in the config cells to match the real GPS location of each image, then run all cells.

---

## Batch inference — CLI scripts

Both scripts must be run from the `clip_surgery/` directory so that `_path_setup.py` is importable, or with that directory on `PYTHONPATH`.

### GeoCLIP — `inference_geoclip.py`

Computes surgery maps for a collection of RGB images grouped by place.

**Input data layout**

```
<source_folder>/
└── <place_name>/
    ├── index.csv          # columns: id, lon, lat
    └── images/
        ├── img_001.jpg
        └── img_002.jpg
```

`index.csv` maps each image filename to GPS coordinates:

```csv
id,lon,lat
img_001.jpg,2.3522,48.8566
img_002.jpg,2.3510,48.8540
```

A second global CSV (`--csv_path`) with columns `IMG_ID`, `LON`, `LAT` provides the full set of candidate locations used to build the location-feature matrix.

**Usage**

```bash
cd src/explainability/clip_surgery

python inference_geoclip.py \
    --csv_path /data/im2gps300/im2gps_places365.csv \
    --source_folder /data/cities_rgb \
    [--no_surgery]       # disable surgery (plain CLIP similarity)
    [--no_layer_grid]    # skip per-layer grid; save only the final-layer map
    [--normalize minmax] # heatmap scaling: "none" | "minmax" (default)
```

**Output** — written to `out/geoclip/<place_name>/<image_id>/`:

| File | Content |
|---|---|
| `layers.png` | Per-layer similarity grid (last 6 encoder layers) |
| `sum_layers.png` | Sum of last-6-layer heatmaps, normalised |

---

### SatCLIP — `inference_satclip.py`

Computes surgery maps for Sentinel-2 GeoTIFF tiles grouped by place.

**Input data layout**

```
<source_folder>/
└── <place_name>/
    ├── index.csv             # columns: id, lon, lat  (id = .tif filename)
    ├── data/
    │   ├── tile_001.tif      # Sentinel-2 GeoTIFF, 12 or 13 bands
    │   └── tile_002.tif
    └── images_corr/          # optional RGB previews (used as overlay background)
        ├── tile_001.png
        └── tile_002.png
```

`index.csv` example:

```csv
id,lon,lat
tile_001.tif,6.9593,50.3752
tile_002.tif,6.9610,50.3800
```

If `images_corr/` is absent the script falls back to `images/`; if neither exists the overlay background is derived from the `.tif` bands directly.

**Usage**

```bash
cd src/explainability/clip_surgery

python inference_satclip.py \
    --source_folder /data/cities50 \
    [--no_surgery]
    [--no_layer_grid]
    [--normalize minmax]
    [--bbox]                        # detect salient regions on the summed heatmap
    [--bbox_percentile 0.9]         # quantile threshold for the saliency mask
    [--bbox_min_peak_fraction 0.5]  # min fraction of tile peak to keep
    [--bbox_max_area 200]           # max connected-component area in pixels
    [--bbox_open_kernel 3]          # morphological opening kernel size (odd ≥ 3; 0 = off)
    [--bbox_square_side 32]         # fixed square side for every bounding box
    [--bbox_line_thickness 2]       # bounding-box line thickness
```

**Output** — written to `out/satclip/<place_name>/<tile_id>/`:

| File | Content |
|---|---|
| `layers.png` | Per-layer similarity grid (last 6 encoder blocks) |
| `sum_layers.png` | Sum of last-6-layer heatmaps, normalised |
| `bbox.png` | Overlay with detected bounding boxes (only with `--bbox`) |
| `mask.png` | Binary saliency mask (only with `--bbox`) |

---

# Surgery Vision Transformer — what changes from a traditional ViT

 Concretely, four things differ from a "vanilla" timm ViT for CLIP-Surgery.

## 1. Last 6 blocks use a different attention (`ConsistentAttention`)

In a normal ViT, every block uses the same `Attention` class. Here, while building blocks, the last `min(6, depth)` blocks are forced to `consistent_attn`:

```python
dpr = calculate_drop_path_rates(drop_path_rate, depth)  # stochastic depth decay rule
self.blocks = nn.ModuleList()
for i in range(depth):
    # apply architecture surgery on the last 6 blocks
    start_index = max(0, depth - 6)
    apply_surgery = i >= start_index
    attn_layer_i = "consistent_attn" if apply_surgery else attn_layer
    self.blocks.append(
        block_fn(
            ...
            attn_layer=attn_layer_i,
            depth=i,
            ...
        )
    )
```

`ConsistentAttention` is the CLIP-Surgery "v-v" (a.k.a. "consistent self-attention") variant from `images/clip_surgery/modified_attention.py` and is wired in via the `ATTN_LAYERS` registry:

```python
ATTN_LAYERS = {
    "": Attention,
    "attn": Attention,
    "consistent_attn": ConsistentAttention,
    # "diff": DiffAttention,
}
```

## 2. The block forward becomes dual-path

A normal ViT block is:

```text
x = x + drop_path(ls(attn(norm(x))))
x = x + drop_path(ls(mlp(norm(x))))
```

Here, when the block uses `ConsistentAttention`, attention returns **two outputs** `(x_attn, x_ori_attn)` and the block routes them as two parallel streams:

```python
def forward(self, x, attn_mask=None):
    if isinstance(self.attn, ConsistentAttention):
        if isinstance(x, list):
            x, x_ori = x
            x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x_ori), attn_mask=attn_mask)))
            x_ori += x_ori_attn
            x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
            x_ori += x_ori_ffn

            x += x_attn  # skip ffn for the new path
            return [x, x_ori]
        else:
            x_attn, x_ori_attn = self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
            x_ori = x + x_ori_attn
            x_ori_ffn = self.drop_path2(self.ls2(self.mlp(self.norm2(x_ori))))
            x_ori += x_ori_ffn
            x += x_attn  # skip ffn for the new path
            return [x, x_ori]
    else:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x
```

Two things to note:

- `x_ori` is the **standard** ViT path (residual attn + residual MLP).
- `x_new` (the one named `x` in the code) gets only the new attention residual and **skips the MLP**.
- Pre-norms (`norm1`, `norm2`) are always applied to `x_ori`, not the new path.

## 3. End-of-backbone merge (CLS from original path, tokens from new path)

After all blocks run, the dual streams are merged so that the final token tensor has the **CLS token from the original stream** but the **patch tokens from the new stream**:

```python
if attn_mask is not None:
    for blk in self.blocks:
        x = blk(x, attn_mask=attn_mask)
elif self.grad_checkpointing and not torch.jit.is_scripting():
    x = checkpoint_seq(self.blocks, x)
else:
    x = self.blocks(x)

# x is a list [x_new, x_ori] because last blocks use ConsistentAttention
if isinstance(x, list) and len(x) == 2:
    x_new, x_ori = x
    x_new[:, 0, :] = x_ori[:, 0, :]  # cls from original path, img tokens from new path
    x = x_new

x = self.norm(x)
```

The same merge is reused inside `forward_intermediates` via a small helper:

```python
def _merge_satclip_dual_path_nlc(x):
    if isinstance(x, list) and len(x) == 2:
        x_new, x_ori = x
        out = x_new.clone()
        out[:, 0, :] = x_ori[:, 0, :]
        return out
    return x
```

A vanilla ViT just runs `x = self.norm(self.blocks(x))` with no list merging.

## 4. `forward_intermediates` is dual-path-aware

The intermediates API exists in upstream `timm` too, but here it’s extended to merge the dual path before saving each layer's output:

```python
if i in take_indices:
    # Dual-path blocks return [x_new, x_ori]; merge before norm / storage.
    xm = _merge_satclip_dual_path_nlc(x)
    # Must clone: later ConsistentAttention blocks mutate `x` / list paths in-place;
    # shared refs would make every saved "layer" converge to the final activations.
    to_store = self.norm(xm) if norm else xm
    intermediates.append(to_store.detach().clone())
```

---

# GeoCLIP CLIP-Surgery — `geoclip_surgery/`

GeoCLIP uses the HuggingFace **Transformers** CLIP vision model, not timm, so the surgery is implemented by subclassing Transformers' own CLIP classes rather than patching a timm ViT. The logic mirrors SatCLIP surgery but the code structure reflects the Transformers API.

## 1. V-V ("consistent") attention — `CLIPSurgeryAttention`

Standard CLIP attention computes `Attention(Q, K, V)` once. `CLIPSurgeryAttention` runs two attention passes in a single forward call:

```python
# --- original path ---
attn_output_ori, _ = attention_interface(self, queries, keys, values, ...)

# --- surgery path: replace Q and K with V (v-v attention) ---
keys = values
queries = keys
attn_output, _ = attention_interface(self, queries, keys, values, ...)
```

Both outputs are projected through the same `out_proj` and returned as a pair:

```python
return [(attn_output, attn_weights), (attn_output_ori, attn_weights_ori)]
```

## 2. Last 6 layers use surgery — `CLIPSurgeryEncoder`

Only the last `min(6, depth)` encoder layers are swapped to `CLIPSurgeryEncoderLayer`; earlier layers remain standard `CLIPEncoderLayer`:

```python
for i in range(depth):
    start_index = max(0, depth - 6)
    apply_surgery = i >= start_index
    attn_layer_i = CLIPSurgeryEncoderLayer if apply_surgery else CLIPEncoderLayer
    layers.append(attn_layer_i(config))
```

## 3. Dual-path block forward — `CLIPSurgeryEncoderLayer`

Once the first surgery layer is entered, `hidden_states` becomes a two-element list `[h_surgery, h_ori]` that propagates through all remaining surgery layers:

```python
# standard (original) path — residual attn + residual MLP
hidden_states_ori = residual_ori + attn_output_ori
hidden_states_ori = residual_ori + mlp(norm2(hidden_states_ori))

# surgery path — residual attn only; MLP is skipped
hidden_states = residual_surgery + attn_output

return [hidden_states, hidden_states_ori]
```

Pre-norms (`layer_norm1`, `layer_norm2`) are always applied to the **original** path; the surgery path only receives the attention residual.

## 4. End-of-encoder merge

After all layers, the two streams are merged the same way as SatCLIP: **CLS token from the original path, patch tokens from the surgery path**:

```python
if isinstance(hidden_states, list) and len(hidden_states) == 2:
    hidden_states, hidden_states_ori = hidden_states
    hidden_states[:, 0, :] = hidden_states_ori[:, 0, :]
```

Because the CLS token (used as GeoCLIP's pooled output) is preserved from the original path, GeoCLIP inference with surgery is numerically identical to the unmodified model. Only the patch tokens differ, and those are used exclusively for visualisation.

## 5. `forward_intermediates` on `CLIPSurgeryVisionTransformer`

`CLIPSurgeryVisionTransformer` adds `forward_intermediates` (absent in the base Transformers `CLIPVisionModel`) so the inference script can capture per-layer hidden states for the layer-grid visualisation:

```python
encoder_outputs, intermediates = self.encoder.forward_intermediates(
    inputs_embeds=hidden_states, indices=indices, ...
)
```

Each recorded intermediate is the merged NLD tensor (`.clone()`) **before** `post_layernorm`, consistent with how per-layer heatmaps are computed in `inference_geoclip.py`.

## 6. Loading with `get_geoclip()`

`geoclip_surgery/load.py` hot-swaps the vision model inside a standard GeoCLIP instance:

```python
model = GeoCLIP()
if surgery:
    config = model.image_encoder.CLIP.vision_model.config   # reuse existing config
    orig_state = model.image_encoder.CLIP.vision_model.state_dict()
    surgery_vision_model = CLIPSurgeryVisionTransformer(config)
    surgery_vision_model.load_state_dict(orig_state, strict=False)  # surgery attn shares proj weights
    model.image_encoder.CLIP.vision_model = surgery_vision_model
```

\\

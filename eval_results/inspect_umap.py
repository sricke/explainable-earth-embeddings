#!/usr/bin/env python3
"""Quick script to render the UMAP and save to PNG for inspection.
UMAP coords are cached to /tmp/umap_cache.npy so plot iterations are fast.
"""
import json, sys
from pathlib import Path

sys.path.append("..")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors

from paths import load_paths
from plot import (
    load_model, load_locations, sample_locations,
    embed_locations, embed_concepts,
    center_renorm, get_continents, umap_reduce,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "satclip_open_clip_vit_l"
CONCEPT_DATASET = "git-10m"
CONCEPT_NAME   = "geospatial_all"
NUM_SAMPLES    = 10_000
STRATIFY       = True
LAT_COL, LON_COL = "lat", "lon"
LOCATIONS_PATH = Path("/home/libe2152/data/dense_grid/dense_grid.csv")
OUT_PNG        = Path("/tmp/umap_inspect.png")
CACHE          = Path("/tmp/umap_cache.npz")

_paths     = load_paths()
MODEL_PATH = str(_paths["models"][MODEL_NAME]["model_path"])
CONCEPT_SET = str(_paths["concept_sets"][CONCEPT_DATASET][CONCEPT_NAME])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load / cache UMAP coords ──────────────────────────────────────────────────
if CACHE.exists():
    print("Loading cached UMAP coords …")
    cache = np.load(CACHE, allow_pickle=True)
    Y_umap_loc = cache["loc"]
    Y_umap_con = cache["con"]
    cont_labels = list(cache["cont_labels"])
    concepts    = list(cache["concepts"])
else:
    print("Computing embeddings + UMAP (will be cached) …")
    df = load_locations(LOCATIONS_PATH, MODEL_PATH)
    df = sample_locations(df, NUM_SAMPLES, STRATIFY)

    loc_precomputed = "location_embedding" in df.columns
    model, _ = load_model(MODEL_PATH, device, loc_precomputed=loc_precomputed)

    loc_emb = embed_locations(model, df, device, lat_col=LAT_COL, lon_col=LON_COL)
    concepts_raw = json.loads(Path(CONCEPT_SET).read_text())
    concepts = list(concepts_raw.keys()) if isinstance(concepts_raw, dict) else [str(c) for c in concepts_raw]
    concept_emb = embed_concepts(model, concepts)

    loc_mc = center_renorm(loc_emb, center=True)
    con_mc = center_renorm(concept_emb, center=True)
    cont_labels, _ = get_continents(df)

    combined = torch.cat([loc_mc, con_mc]).numpy()
    n = len(loc_mc)
    Y_umap = umap_reduce(combined, n_components=2, n_neighbors=200, min_dist=0.8, metric="cosine", pca_dim=50)
    Y_umap_loc, Y_umap_con = Y_umap[:n], Y_umap[n:]

    np.savez(CACHE,
             loc=Y_umap_loc, con=Y_umap_con,
             cont_labels=np.array(cont_labels, dtype=object),
             concepts=np.array(concepts, dtype=object))
    print("Cached.")

# ── Palette & style ───────────────────────────────────────────────────────────
CONTINENT_PALETTE = {
    "Africa":        "#E8761B",
    "Asia":          "#D62728",
    "Europe":        "#2CA02C",
    "North America": "#1F77B4",
    "South America": "#9467BD",
    "Oceania":       "#17BECF",
    "Antarctica":    "#BCBD22",
    "Unknown":       "#999999",   # darker so it reads on white
}
CONCEPT_COLOR = "#E91C8E"   # vivid magenta — distinct from all continent colours

RC_PARAMS = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "axes.titlesize":     14,
    "axes.labelsize":     12,
    "font.size":          12,
    "legend.fontsize":    11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "axes.linewidth":     0.8,
    "figure.dpi":         150,
    "savefig.dpi":        150,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.08,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
}

# ── Drawing helpers ───────────────────────────────────────────────────────────
def label_overlapping_concepts(ax, Y_loc, Y_con, names, radius=0.25, min_neighbors=20, jitter=0.4):
    nn = NearestNeighbors(radius=radius).fit(Y_loc)
    counts = np.array([len(nn.radius_neighbors([p], return_distance=False)[0]) for p in Y_con])
    for i, cnt in enumerate(counts):
        if cnt < min_neighbors:
            continue
        x, y = Y_con[i]
        dx = ((np.sin(i * 12.9898) * 43758.5453) % 1 - 0.5) * jitter
        dy = ((np.sin(i * 78.233)  * 12345.6789) % 1 - 0.5) * jitter
        ax.text(x + dx - 0.2, y + dy + 0.5, names[i],
                fontsize=8, ha="center", va="bottom", zorder=20,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def draw_panel(ax, Y_loc, Y_con, cont_labels, loc_size=9, con_size=75, loc_alpha=0.6):
    ca = np.array(cont_labels)
    order = ["Unknown"] + [c for c in CONTINENT_PALETTE if c != "Unknown" and c in set(ca)]
    for cont in order:
        m = ca == cont
        if not m.any():
            continue
        ax.scatter(Y_loc[m, 0], Y_loc[m, 1], s=loc_size, marker="o", linewidths=0,
                   c=[CONTINENT_PALETTE[cont]], alpha=loc_alpha, rasterized=True)
    # shadow
    ax.scatter(Y_con[:, 0] + 0.05, Y_con[:, 1] - 0.05,
               s=con_size * 1.15, marker="^", color="black", alpha=0.4,
               linewidths=0, zorder=4, rasterized=True)
    ax.scatter(Y_con[:, 0], Y_con[:, 1],
               s=con_size, marker="^", zorder=6,
               facecolors=CONCEPT_COLOR, edgecolors="white", linewidths=0.1, rasterized=True)

    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    all_pts = np.concatenate([Y_loc, Y_con])
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim]):
        lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
        pad = 0.04 * (hi - lo)
        setter(lo - pad, hi + pad)
    label_overlapping_concepts(ax, Y_loc, Y_con, names=concepts)


def _mk_handle(label, marker, color, size=7, edgecolor="None", edgewidth=0):
    return Line2D([0], [0], marker=marker, linestyle="None", markersize=size,
                  markerfacecolor=color, markeredgecolor=edgecolor,
                  markeredgewidth=edgewidth, label=label)


def add_legend(ax, fig, present, bottom=0.22):
    type_handles = [
        _mk_handle("Location", "o", "#888", size=6),
        _mk_handle("Concept",  "^", CONCEPT_COLOR, size=8, edgecolor="white", edgewidth=0.5),
    ]
    leg_types = ax.legend(
        handles=type_handles, title="Point type", title_fontsize=8,
        loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc",
        fontsize=9, handletextpad=0.5, borderpad=0.6, alignment="center",
    )
    ax.add_artist(leg_types)

    cont_handles = [_mk_handle(c, "o", CONTINENT_PALETTE[c], size=7) for c in present]
    ncol = min(len(cont_handles), 4)
    fig.legend(
        handles=cont_handles, title="Region", title_fontsize=8,
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol,
        frameon=False, fontsize=9, columnspacing=1.2, handletextpad=0.4,
        alignment="center",
    )
    fig.subplots_adjust(bottom=bottom)


# ── Render ────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update(RC_PARAMS)
fig, ax = plt.subplots(figsize=(8, 7))
draw_panel(ax, Y_umap_loc, Y_umap_con, cont_labels)
present = [c for c in CONTINENT_PALETTE if c in set(cont_labels)]
add_legend(ax, fig, present)
fig.savefig(OUT_PNG)
print(f"Saved → {OUT_PNG}")

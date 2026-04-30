#!/usr/bin/env python3
"""Camera-ready UMAP / t-SNE plots of location and concept embeddings."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap
import geopandas as gpd
from shapely.geometry import Point
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append("..")
sys.path.append("../..")
from models.model import build_model
from models.finetune import apply_lora

SHAPEFILE = Path.home() / "data/shapefiles/ne_110m_admin_0_countries.shp"

CONTINENT_PALETTE = {
    "Africa":        "#4C78A8",  # cool
    "Asia":          "#72B7B2",  # cool
    "Europe":        "#54A24B",  # cool
    "North America": "#1F77B4",  # cool
    "South America": "#6BAED6",  # cool
    "Oceania":       "#9ECAE1",  # cool
    "Antarctica":    "#FFA500",  # warm
    "Unknown":       None,       # will be filtered out
}
_KNOWN_CONTINENTS = frozenset(CONTINENT_PALETTE) - {"Unknown"}
CONCEPT_COLOR = "#E4572E"

RC_PARAMS = {
    "font.family":        "serif",
    "font.serif":         ["Linux Libertine O", "Libertinus Serif", "Times New Roman",
                           "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
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
    "savefig.dpi":        400,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
}


def _alpha(c, a):
    r, g, b, _ = mcolors.to_rgba(c)
    return r, g, b, a


def _with_alpha(color, alpha):
    return _alpha(color, alpha)


def load_model(ckpt_path, device, loc_precomputed=True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    model = build_model(
        text_encoder=a.text_encoder,
        location_encoder=a.location_encoder,
        text_projection=a.text_projection,
        location_projection=a.location_projection,
        shared_dim=a.shared_dim,
        text_finetune_mode=a.text_finetune_mode,
        loc_finetune_mode=a.loc_finetune_mode,
        text_proj_hidden_layers=a.text_proj_hidden_layers,
        text_proj_hidden_features=a.text_proj_hidden_features,
        loc_proj_hidden_layers=a.loc_proj_hidden_layers,
        loc_proj_hidden_features=a.loc_proj_hidden_features,
        text_nonlinearity=a.text_nonlinearity,
        loc_nonlinearity=a.loc_nonlinearity,
        precomputed_text_embeddings=False,
        precomputed_location_embeddings=loc_precomputed,
        device=device,
    )
    if a.text_finetune_mode == "lora":
        model.text_encoder.text_encoder.m = apply_lora(model.text_encoder.text_encoder.m, a.lora_rank)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), a


@torch.no_grad()
def embed_locations(model, df, device, lat_col="lat", lon_col="lon", batch_size=4096):
    if model.location_encoder.precomputed:
        embs = torch.from_numpy(np.array(df["location_embedding"].tolist(), dtype=np.float32).copy())
    else:
        embs = torch.from_numpy(df[[lat_col, lon_col]].to_numpy(dtype=np.float32, copy=True))
    return torch.cat([model.location_model_predict(b.to(device))
                      for b in tqdm(DataLoader(embs, batch_size), desc="Loc emb")]).cpu()


@torch.no_grad()
def embed_concepts(model, concepts, batch_size=4096):
    return torch.cat([model.text_model_predict(concepts[i:i+batch_size])
                      for i in tqdm(range(0, len(concepts), batch_size), desc="Concept emb")]).cpu()


def center_renorm(X, center=False):
    X = F.normalize(X, dim=1)
    return F.normalize(X - X.mean(0, keepdim=True), dim=1) if center else X


def umap_reduce(X, n_components=2, pca_dim=None, n_neighbors=15, min_dist=0.1, metric="cosine"):
    if pca_dim:
        p = min(pca_dim, X.shape[1], X.shape[0] - 1)
        if p >= 2:
            X = PCA(n_components=p, random_state=0).fit_transform(X)
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                     metric=metric, random_state=0, n_jobs=1).fit_transform(X)


def tsne_reduce(X, n_components=2, perplexity=30, learning_rate=200, max_iter=1000):
    return TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                max_iter=max_iter, init="pca", metric="cosine", random_state=0).fit_transform(X)


def get_continents(df):
    world = gpd.read_file(SHAPEFILE)
    col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=[Point(lon, lat) for lat, lon in zip(df.lat, df.lon)],
                           crs="EPSG:4326")
    joined = gpd.sjoin(gdf, world[[col, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    labels = [c if c in _KNOWN_CONTINENTS else "Unknown" for c in joined[col].fillna("Unknown")]
    return labels, [CONTINENT_PALETTE[c] for c in labels]


def _draw_panel(ax, Y_loc, Y_con, cont_labels, title=None,
                loc_size=9, con_size=85, loc_alpha=0.6):
    ca = np.array(cont_labels)
    order = ["Unknown"] + [c for c in CONTINENT_PALETTE if c != "Unknown" and c in set(ca)]
    for cont in order:
        m = ca == cont
        if not m.any():
            continue
        ax.scatter(Y_loc[m, 0], Y_loc[m, 1], s=loc_size, marker="o",
                   c=[_alpha(CONTINENT_PALETTE[cont], loc_alpha)],
                   linewidths=0, rasterized=True)
    ax.scatter(Y_con[:, 0], Y_con[:, 1], s=con_size, marker="^", zorder=5,
               facecolors=CONCEPT_COLOR, edgecolors="white", linewidths=0.7,
               rasterized=True)

    if title is not None:
        ax.set_title(title, pad=8, fontweight="regular")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    ax_x = np.concatenate([Y_loc[:, 0], Y_con[:, 0]])
    ax_y = np.concatenate([Y_loc[:, 1], Y_con[:, 1]])
    xr, yr = ax_x.ptp(), ax_y.ptp()
    ax.set_xlim(ax_x.min() - 0.04 * xr, ax_x.max() + 0.04 * xr)
    ax.set_ylim(ax_y.min() - 0.04 * yr, ax_y.max() + 0.04 * yr)


def _legend_handles(present):
    cont = [Line2D([0], [0], marker="o", linestyle="None", markersize=7,
                   markerfacecolor=CONTINENT_PALETTE[c], markeredgewidth=0, label=c)
            for c in present]
    types = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor="#777", markeredgewidth=0, label="Location"),
        Line2D([0], [0], marker="^", linestyle="None", markersize=8,
               markerfacecolor=CONCEPT_COLOR, markeredgecolor="white",
               markeredgewidth=0.7, label="Concept"),
    ]
    return cont, types


def plot_png(Y_loc, Y_con, cont_labels, method, out_path):
    matplotlib.rcParams.update(RC_PARAMS)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    _draw_panel(ax, Y_loc, Y_con, cont_labels, title=method)

    present = [c for c in CONTINENT_PALETTE if c in set(cont_labels)]
    cont, types = _legend_handles(present)
    handles = cont + types
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=min(len(handles), 5), frameon=False,
               columnspacing=1.4, handletextpad=0.4)
    fig.subplots_adjust(bottom=0.18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")


def plot_panel(panels, cont_labels, out_path):
    matplotlib.rcParams.update(RC_PARAMS)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 5.4))
    if n == 1:
        axes = [axes]
    for ax, (Y_loc, Y_con, method) in zip(axes, panels):
        _draw_panel(ax, Y_loc, Y_con, cont_labels, title=method)

    present = [c for c in CONTINENT_PALETTE if c in set(cont_labels)]
    cont, types = _legend_handles(present)
    handles = cont + types
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(handles), frameon=False,
               columnspacing=1.3, handletextpad=0.4)
    fig.subplots_adjust(bottom=0.13, wspace=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved PNG: {out_path}")


def plot_html(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels,
              loc_labels, concept_labels, method, out_path):
    ca = np.array(cont_labels)
    conts = sorted(set(cont_labels))

    def build(Y_loc, Y_txt, visible):
        traces = []
        for cont in conts:
            idx = np.where(ca == cont)[0]
            traces.append(go.Scattergl(
                x=Y_loc[idx, 0], y=Y_loc[idx, 1], mode="markers",
                marker=dict(size=4, color=CONTINENT_PALETTE.get(cont, "#999"), opacity=0.6),
                name=cont, legendgroup=cont, showlegend=visible,
                text=[f"{loc_labels[i]}<br>{cont}" for i in idx],
                hovertemplate="%{text}<extra></extra>", visible=visible))
        traces.append(go.Scatter(
            x=Y_txt[:, 0], y=Y_txt[:, 1], mode="markers",
            marker=dict(size=11, color=CONCEPT_COLOR, opacity=0.95, symbol="triangle-up",
                        line=dict(width=1, color="white")),
            name="Concept", legendgroup="concept", showlegend=visible,
            hovertext=concept_labels, hovertemplate="%{hovertext}<extra></extra>",
            visible=visible))
        return traces

    raw = build(Y_loc_raw, Y_txt_raw, True)
    mc  = build(Y_loc_mc,  Y_txt_mc,  False)
    fig = go.Figure(data=raw + mc)
    n = len(raw)
    vis_raw = [True] * n + [False] * n
    vis_mc  = [False] * n + [True]  * n

    fig.update_layout(
        title=dict(text=f"{method} — raw", font_size=15),
        width=1200, height=750, template="plotly_white", hovermode="closest",
        legend=dict(x=1.01, y=1.0, xanchor="left", font_size=11),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.0, y=1.08,
            xanchor="left", yanchor="top", showactive=True,
            buttons=[
                dict(label="Raw", method="update",
                     args=[{"visible": vis_raw}, {"title.text": f"{method} — raw"}]),
                dict(label="Mean-centred", method="update",
                     args=[{"visible": vis_mc}, {"title.text": f"{method} — mean-centred"}]),
            ])])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn",
                   config={"scrollZoom": True, "displayModeBar": True})
    print(f"Saved HTML: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--concept_set", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--stratify_by_continent", action="store_true")
    p.add_argument("--locations_path", default=None)
    p.add_argument("--lat_col", default="lat")
    p.add_argument("--lon_col", default="lon")
    p.add_argument("--combined", action="store_true",
                   help="Save a single side-by-side UMAP+t-SNE figure as well.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else Path("plots") / Path(args.concept_set).stem

    if args.locations_path is not None:
        loc_p = Path(args.locations_path)
        files = sorted(loc_p.glob("*.parquet")) or sorted(loc_p.glob("*.csv")) if loc_p.is_dir() else [loc_p]
        dfs = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    else:
        ckpt_dataset = argparse.Namespace(
            **torch.load(args.model_path, map_location="cpu", weights_only=False)["args"]).dataset_path
        dfs = [pd.read_parquet(p_) for p_ in sorted(Path(ckpt_dataset).glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)

    loc_precomputed = "location_embedding" in df.columns
    model, _ = load_model(args.model_path, device, loc_precomputed=loc_precomputed)

    if args.num_samples and len(df) > args.num_samples:
        if args.stratify_by_continent:
            cont_full, _ = get_continents(df)
            df["_continent"] = cont_full
            n_per = max(1, args.num_samples // df["_continent"].nunique())
            df = (df.groupby("_continent", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), n_per), random_state=0))
                    .reset_index(drop=True))
        df = (df.sample(frac=1, random_state=0).head(args.num_samples)
                .drop(columns="_continent", errors="ignore").reset_index(drop=True))

    loc_emb = embed_locations(model, df, device, lat_col=args.lat_col, lon_col=args.lon_col)
    concepts = json.loads(Path(args.concept_set).read_text())
    concepts = list(concepts.keys()) if isinstance(concepts, dict) else [str(c) for c in concepts]
    concept_emb = embed_concepts(model, [f"a satellite image of {c}" for c in concepts])

    loc_raw, con_raw = center_renorm(loc_emb), center_renorm(concept_emb)
    loc_mc,  con_mc  = center_renorm(loc_emb, True), center_renorm(concept_emb, True)
    cont_labels, _ = get_continents(df)
    loc_labels = [f"{r[args.lat_col]:.4f}, {r[args.lon_col]:.4f}"
                  for _, r in df[[args.lat_col, args.lon_col]].iterrows()]

    n = len(loc_raw)
    panels = []
    for method, reducer in [("UMAP", umap_reduce), ("t-SNE", tsne_reduce)]:
        name = method.lower().replace("-", "")
        Y_raw = reducer(torch.cat([loc_raw, con_raw]).numpy())
        Y_mc  = reducer(torch.cat([loc_mc,  con_mc ]).numpy())
        plot_png(Y_mc[:n], Y_mc[n:], cont_labels, method, out_dir / f"{name}_mean_centered.png")
        plot_html(Y_raw[:n], Y_mc[:n], Y_raw[n:], Y_mc[n:],
                  cont_labels, loc_labels, concepts, method, out_dir / f"{name}.html")
        panels.append((Y_mc[:n], Y_mc[n:], method))

    if args.combined:
        plot_panel(panels, cont_labels, out_dir / "combined_mean_centered.png")


if __name__ == "__main__":
    main()

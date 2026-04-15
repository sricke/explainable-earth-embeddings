#!/usr/bin/env python3
"""Plot UMAP visualizations (PNG + interactive HTML) of location and concept embeddings."""
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import umap
import geopandas as gpd

from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append("..")
sys.path.append("../..")
from models.model import TextLocationModel
from models.utils import make_text_encoder, make_location_encoder

SHAPEFILE = Path.home() / "data/shapefiles/ne_110m_admin_0_countries.shp"
CONTINENT_PALETTE = {
    "Africa": "#e07b39", "Antarctica": "#88ccee", "Asia": "#cc3333",
    "Europe": "#4477aa", "North America": "#ff9900", "Oceania": "#ee77aa",
    "South America": "#aa44aa", "Unknown": "#aaaaaa",
}


def load_model(ckpt_path, device, loc_precomputed=True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    text_enc = make_text_encoder(
        a.text_encoder, a.text_projection, a.shared_dim, a.text_finetune_mode,
        num_hidden_layers=a.text_proj_hidden_layers,
        num_hidden_features=a.text_proj_hidden_features,
        nonlinearity=a.text_nonlinearity, precomputed=False,
    )
    loc_enc = make_location_encoder(
        a.location_encoder, a.location_projection, a.shared_dim, a.loc_finetune_mode,
        num_hidden_layers=a.loc_proj_hidden_layers,
        num_hidden_features=a.loc_proj_hidden_features,
        nonlinearity=a.loc_nonlinearity, precomputed=loc_precomputed,
    )
    model = TextLocationModel(text_enc, loc_enc).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), a


@torch.no_grad()
def embed_locations(model, df, device, lat_col="lat", lon_col="lon", batch_size=4096):
    if model.location_encoder.precomputed:
        # Some parquet/arrow paths can produce arrays with negative strides; force a contiguous copy for torch.
        embs_np = np.array(df["location_embedding"].tolist(), dtype=np.float32).copy()
        embs = torch.from_numpy(embs_np)
    else:
        # pandas can also yield negative-stride views; force a contiguous copy for torch.
        embs_np = df[[lat_col, lon_col]].to_numpy(dtype=np.float32, copy=True)
        embs = torch.from_numpy(embs_np)
    out = [model.location_model(b.to(device)) for b in tqdm(DataLoader(embs, batch_size), desc="Loc emb")]
    return torch.cat(out).cpu()


@torch.no_grad()
def embed_concepts(model, concepts, batch_size=4096):
    out = []
    for i in tqdm(range(0, len(concepts), batch_size), desc="Concept emb"):
        out.append(model.text_model_predict(concepts[i:i + batch_size]))
    return torch.cat(out).cpu()


def center_renorm(X, center=False):
    X = F.normalize(X, dim=1)
    return F.normalize(X - X.mean(0, keepdim=True), dim=1) if center else X


def umap_2d(X_np, pca_dim=None, n_neighbors=50, min_dist=0.1, metric="cosine"):
    if pca_dim is not None:
        p = min(pca_dim, X_np.shape[1], X_np.shape[0] - 1)
        if p >= 2:
            X_np = PCA(n_components=p, random_state=0).fit_transform(X_np)
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=0,
    ).fit_transform(X_np)

def get_continents(df):
    world = gpd.read_file(SHAPEFILE)
    col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(lon, lat) for lat, lon in zip(df.lat, df.lon)], crs="EPSG:4326")
    joined = gpd.sjoin(gdf, world[[col, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    labels = joined[col].fillna("Unknown").tolist()
    return labels, [CONTINENT_PALETTE.get(c, "#aaaaaa") for c in labels]


def plot_png(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, cont_colors, out_path):
    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    ca, cc = np.array(cont_labels), np.array(cont_colors)
    for ax, Yl, Yt, title in zip(axes, [Y_loc_raw, Y_loc_mc], [Y_txt_raw, Y_txt_mc], ["raw", "mean-centred"]):
        for cont in sorted(set(cont_labels)):
            m = ca == cont
            ax.scatter(Yl[m, 0], Yl[m, 1], s=6, alpha=0.45, color=cc[m][0], label=cont)
        ax.scatter(Yt[:, 0], Yt[:, 1], s=6, alpha=0.6, color="tab:green", marker="^", label="Concepts")
        ax.set_title(f"UMAP — {title}")
        ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved PNG: {out_path}")


def plot_html(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, loc_labels, concept_labels, out_path):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("UMAP — raw", "UMAP — mean-centred"), horizontal_spacing=0.08)
    ca = np.array(cont_labels)

    def add_loc(Y, col):
        for cont in sorted(set(cont_labels)):
            idx = np.where(ca == cont)[0]
            fig.add_trace(go.Scatter(
                x=Y[idx, 0], y=Y[idx, 1], mode="markers",
                marker=dict(size=4, color=CONTINENT_PALETTE.get(cont, "#aaaaaa"), opacity=0.6),
                name=cont, legendgroup=cont, showlegend=(col == 1),
                text=[f"{loc_labels[i]}<br>{cont}" for i in idx],
                hovertemplate="%{text}<extra></extra>",
            ), row=1, col=col)

    def add_txt(Y, col):
        fig.add_trace(go.Scatter(
            x=Y[:, 0], y=Y[:, 1], mode="markers",
            marker=dict(size=5, color="#16a34a", opacity=0.75, symbol="diamond"),
            name="Concepts", legendgroup="concepts", showlegend=(col == 1),
            text=concept_labels, hovertemplate="%{text}<extra></extra>",
        ), row=1, col=col)

    add_loc(Y_loc_raw, 1); add_txt(Y_txt_raw, 1)
    add_loc(Y_loc_mc, 2);  add_txt(Y_txt_mc, 2)
    fig.update_layout(width=1400, height=620, template="plotly_white",
                      legend=dict(x=1.02, y=1.0, xanchor="left", font_size=11))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"Saved HTML: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--concept_set", required=True, help="Path to concept JSON file")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: plots/<concept_set_stem>)")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--stratify_by_continent", action="store_true",
                        help="When subsampling, sample equally across continents (requires --num_samples)")
    parser.add_argument("--locations_path", default=None,
                        help="Path to a parquet/CSV file or directory of parquet/CSV files with lat/lon/location_embedding columns. "
                             "Overrides the dataset from the checkpoint.")
    parser.add_argument("--lat_col", default="lat",
                        help="Column name for latitude (used when location_embedding is absent). Default: lat")
    parser.add_argument("--lon_col", default="lon",
                        help="Column name for longitude (used when location_embedding is absent). Default: lon")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) if args.output_dir else Path("plots") / Path(args.concept_set).stem

    if args.locations_path is not None:
        loc_p = Path(args.locations_path)
        files = sorted(loc_p.glob("*.parquet")) or sorted(loc_p.glob("*.csv")) if loc_p.is_dir() else [loc_p]
        dfs = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    else:
        ckpt_dataset = argparse.Namespace(**torch.load(args.model_path, map_location="cpu", weights_only=False)["args"]).dataset_path
        dfs = [pd.read_parquet(p) for p in sorted(Path(ckpt_dataset).glob("*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)

    loc_precomputed = "location_embedding" in df.columns
    model, margs = load_model(args.model_path, device, loc_precomputed=loc_precomputed)

    if args.num_samples is not None and len(df) > args.num_samples:
        if args.stratify_by_continent:
            cont_labels_full, _ = get_continents(df)
            df["_continent"] = cont_labels_full
            continents = df["_continent"].unique()
            n_per_continent = max(1, args.num_samples // len(continents))

            df = (
                df.groupby("_continent", group_keys=False)
                .apply(lambda g: g.sample(
                    n=min(len(g), n_per_continent),  # avoid sampling more than group size
                    random_state=0,
                ))
                .reset_index(drop=True)
            )

        # shuffle + trim in case we slightly exceed num_samples
        df = (
            df.sample(frac=1, random_state=0)
            .head(args.num_samples)
            .drop(columns="_continent", errors="ignore")
            .reset_index(drop=True)
        )

    loc_emb = embed_locations(model, df, device, lat_col=args.lat_col, lon_col=args.lon_col)

    concepts = json.loads(Path(args.concept_set).read_text())
    concepts = list(concepts.keys()) if isinstance(concepts, dict) else [str(c) for c in concepts]
    prompted_concepts = [f"a satellite image of {c}" for c in concepts]
    concept_emb = embed_concepts(model, prompted_concepts)

    loc_raw, con_raw = center_renorm(loc_emb), center_renorm(concept_emb)
    loc_mc,  con_mc  = center_renorm(loc_emb, True), center_renorm(concept_emb, True)

    n = len(loc_raw)
    Y_raw = umap_2d(torch.cat([loc_raw, con_raw]).numpy())
    Y_mc  = umap_2d(torch.cat([loc_mc,  con_mc ]).numpy())
    Y_loc_raw, Y_txt_raw = Y_raw[:n], Y_raw[n:]
    Y_loc_mc,  Y_txt_mc  = Y_mc[:n],  Y_mc[n:]

    cont_labels, cont_colors = get_continents(df)
    loc_labels = [f"{row[args.lat_col]:.4f}, {row[args.lon_col]:.4f}" for _, row in df[[args.lat_col, args.lon_col]].iterrows()]

    plot_png(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, cont_colors, out_dir / "umap.png")
    plot_html(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, loc_labels, concepts, out_dir / "umap.html")


if __name__ == "__main__":
    main()

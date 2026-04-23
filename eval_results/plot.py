#!/usr/bin/env python3
"""Plot UMAP/t-SNE visualizations (PNG + interactive 3-D HTML) of location and concept embeddings."""
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.append("..")
sys.path.append("../..")
from models.model import build_model
from models.finetune import apply_lora

SHAPEFILE = Path.home() / "data/shapefiles/ne_110m_admin_0_countries.shp"
CONTINENT_PALETTE = {
    "Africa": "#e07b39", "Antarctica": "#88ccee", "Asia": "#cc3333",
    "Europe": "#4477aa", "North America": "#ff9900", "Oceania": "#ee77aa",
    "South America": "#aa44aa", "Unknown": "#aaaaaa",
}


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
    if a.text_finetune_mode == 'lora':
        model.text_encoder.text_encoder.m = apply_lora(model.text_encoder.text_encoder.m, a.lora_rank)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), a


@torch.no_grad()
def embed_locations(model, df, device, lat_col="lat", lon_col="lon", batch_size=4096):
    if model.location_encoder.precomputed:
        embs = torch.from_numpy(np.array(df["location_embedding"].tolist(), dtype=np.float32).copy())
    else:
        embs = torch.from_numpy(df[[lat_col, lon_col]].to_numpy(dtype=np.float32, copy=True))
    return torch.cat([model.location_model_predict(b.to(device)) for b in tqdm(DataLoader(embs, batch_size), desc="Loc emb")]).cpu()


@torch.no_grad()
def embed_concepts(model, concepts, batch_size=4096):
    return torch.cat([model.text_model_predict(concepts[i:i+batch_size])
                      for i in tqdm(range(0, len(concepts), batch_size), desc="Concept emb")]).cpu()


def center_renorm(X, center=False):
    X = F.normalize(X, dim=1)
    return F.normalize(X - X.mean(0, keepdim=True), dim=1) if center else X


def umap_reduce(X, n_components=3, pca_dim=None, n_neighbors=50, min_dist=0.1):
    if pca_dim:
        p = min(pca_dim, X.shape[1], X.shape[0] - 1)
        if p >= 2:
            X = PCA(n_components=p, random_state=0).fit_transform(X)
    return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                     metric="cosine", random_state=0, n_jobs=1).fit_transform(X)


def tsne_reduce(X, n_components=3, perplexity=30):
    return TSNE(n_components=n_components, perplexity=perplexity, metric="cosine",
                random_state=0).fit_transform(X)


def get_continents(df):
    world = gpd.read_file(SHAPEFILE)
    col = "CONTINENT" if "CONTINENT" in world.columns else "continent"
    gdf = gpd.GeoDataFrame(df.copy(), geometry=[Point(lon, lat) for lat, lon in zip(df.lat, df.lon)], crs="EPSG:4326")
    joined = gpd.sjoin(gdf, world[[col, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    labels = joined[col].fillna("Unknown").tolist()
    return labels, [CONTINENT_PALETTE.get(c, "#aaaaaa") for c in labels]


def plot_png(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, cont_colors, method, out_path):
    _, axes = plt.subplots(1, 2, figsize=(13, 5))
    ca, cc = np.array(cont_labels), np.array(cont_colors)
    for ax, Yl, Yt, ver in zip(axes, [Y_loc_raw, Y_loc_mc], [Y_txt_raw, Y_txt_mc], ["raw", "mean-centred"]):
        for cont in sorted(set(cont_labels)):
            m = ca == cont
            ax.scatter(Yl[m, 0], Yl[m, 1], s=6, alpha=0.45, color=cc[m][0], label=cont)
        ax.scatter(Yt[:, 0], Yt[:, 1], s=6, alpha=0.6, color="tab:green", marker="^", label="Concepts")
        ax.set_title(f"{method} — {ver}")
        ax.legend(markerscale=2, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300); plt.close()
    print(f"Saved PNG: {out_path}")


def plot_html(Y_loc_raw, Y_loc_mc, Y_txt_raw, Y_txt_mc, cont_labels, loc_labels, concept_labels, method, out_path):
    """2-D interactive scatter.  Location points use WebGL (Scattergl) for speed;
    concept points use SVG Scatter so they always render on top with crisp hover."""
    ca = np.array(cont_labels)
    conts = sorted(set(cont_labels))

    traces = []

    # ── Raw traces (initially visible) ──────────────────────────────────────
    for cont in conts:
        idx = np.where(ca == cont)[0]
        traces.append(go.Scattergl(
            x=Y_loc_raw[idx, 0], y=Y_loc_raw[idx, 1], mode="markers",
            marker=dict(size=4, color=CONTINENT_PALETTE.get(cont, "#aaaaaa"), opacity=0.5),
            name=cont, legendgroup=cont, showlegend=True,
            text=[f"{loc_labels[i]}<br>{cont}" for i in idx],
            hovertemplate="%{text}<extra></extra>",
            visible=True,
        ))
    traces.append(go.Scatter(
        x=Y_txt_raw[:, 0], y=Y_txt_raw[:, 1], mode="markers",
        marker=dict(size=11, color="#16a34a", opacity=0.95, symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Concepts", legendgroup="concepts", showlegend=True,
        hovertext=concept_labels, hovertemplate="%{hovertext}<extra></extra>",
        visible=True,
    ))
    n_raw = len(traces)

    # ── Mean-centred traces (initially hidden) ───────────────────────────────
    for cont in conts:
        idx = np.where(ca == cont)[0]
        traces.append(go.Scattergl(
            x=Y_loc_mc[idx, 0], y=Y_loc_mc[idx, 1], mode="markers",
            marker=dict(size=4, color=CONTINENT_PALETTE.get(cont, "#aaaaaa"), opacity=0.5),
            name=cont, legendgroup=cont, showlegend=False,
            text=[f"{loc_labels[i]}<br>{cont}" for i in idx],
            hovertemplate="%{text}<extra></extra>",
            visible=False,
        ))
    traces.append(go.Scatter(
        x=Y_txt_mc[:, 0], y=Y_txt_mc[:, 1], mode="markers",
        marker=dict(size=11, color="#16a34a", opacity=0.95, symbol="diamond",
                    line=dict(width=1, color="white")),
        name="Concepts", legendgroup="concepts", showlegend=False,
        hovertext=concept_labels, hovertemplate="%{hovertext}<extra></extra>",
        visible=False,
    ))
    n_mc = len(traces) - n_raw

    vis_raw = [True]  * n_raw + [False] * n_mc
    vis_mc  = [False] * n_raw + [True]  * n_mc

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"{method} — raw", font_size=15),
        width=1200, height=750,
        template="plotly_white",
        hovermode="closest",
        legend=dict(x=1.01, y=1.0, xanchor="left", font_size=11),
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.0, y=1.08, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Raw",
                     method="update",
                     args=[{"visible": vis_raw}, {"title.text": f"{method} — raw"}]),
                dict(label="Mean-centred",
                     method="update",
                     args=[{"visible": vis_mc}, {"title.text": f"{method} — mean-centred"}]),
            ],
            showactive=True,
        )],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn",
                   config={"scrollZoom": True, "displayModeBar": True})
    print(f"Saved HTML: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--concept_set", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--stratify_by_continent", action="store_true")
    parser.add_argument("--locations_path", default=None)
    parser.add_argument("--lat_col", default="lat")
    parser.add_argument("--lon_col", default="lon")
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
    model, _ = load_model(args.model_path, device, loc_precomputed=loc_precomputed)

    if args.num_samples and len(df) > args.num_samples:
        if args.stratify_by_continent:
            cont_labels_full, _ = get_continents(df)
            df["_continent"] = cont_labels_full
            n_per = max(1, args.num_samples // df["_continent"].nunique())
            df = df.groupby("_continent", group_keys=False).apply(
                lambda g: g.sample(min(len(g), n_per), random_state=0)).reset_index(drop=True)
        df = df.sample(frac=1, random_state=0).head(args.num_samples).drop(columns="_continent", errors="ignore").reset_index(drop=True)

    loc_emb = embed_locations(model, df, device, lat_col=args.lat_col, lon_col=args.lon_col)
    concepts = json.loads(Path(args.concept_set).read_text())
    concepts = list(concepts.keys()) if isinstance(concepts, dict) else [str(c) for c in concepts]
    concept_emb = embed_concepts(model, [f"a satellite image of {c}" for c in concepts])

    loc_raw, con_raw = center_renorm(loc_emb), center_renorm(concept_emb)
    loc_mc,  con_mc  = center_renorm(loc_emb, True), center_renorm(concept_emb, True)
    cont_labels, cont_colors = get_continents(df)
    loc_labels = [f"{row[args.lat_col]:.4f}, {row[args.lon_col]:.4f}" for _, row in df[[args.lat_col, args.lon_col]].iterrows()]

    n = len(loc_raw)
    for method, reducer in [("UMAP", umap_reduce), ("t-SNE", tsne_reduce)]:
        name = method.lower().replace("-", "")
        # 2-D is enough for both the PNG and the interactive HTML
        Y_raw = reducer(torch.cat([loc_raw, con_raw]).numpy(), n_components=2)
        Y_mc  = reducer(torch.cat([loc_mc,  con_mc ]).numpy(), n_components=2)
        plot_png( Y_raw[:n], Y_mc[:n], Y_raw[n:], Y_mc[n:], cont_labels, cont_colors, method, out_dir / f"{name}.png")
        plot_html(Y_raw[:n], Y_mc[:n], Y_raw[n:], Y_mc[n:], cont_labels, loc_labels, concepts, method, out_dir / f"{name}.html")


if __name__ == "__main__":
    main()

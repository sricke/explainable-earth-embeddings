#!/usr/bin/env python3
"""PCA of SatCLIP location embeddings and world-map visualization.

Loads SatCLIP (optional CLIP-Surgery vision wrapper via ``--surgery`` / ``--no-surgery``),
encodes (lon, lat) from a CSV index,
fits PCA on all embeddings, and saves scatter plots on a geographic grid
(equirectangular: x=longitude, y=latitude) colored by each principal component.
Also writes ``world_map_pc123_rgb.png``: **direct** mode blends three colors from ``WORLD_MAP_PC_CMAP``
(Pastel1 by default) weighted by PC1–PC3 after per-PC min–max.
Use ``--pc123-color-mode blend`` for the older blue/yellow/purple additive blend.
By default, maps use a **lon/lat grid** (mean per cell) with **all sample points drawn on top**
(``--map-overlay-points``, default on); use ``--no-map-overlay-points`` for raster only, or
``--no-map-raster`` for scatter only.
World maps omit grid lines and axis frames; when **Cartopy** is installed, land, ocean, and
coastlines are drawn by default (``--no-world-map-continents`` to disable).
World maps use ``--world_map_figsize`` / ``--world_map_dpi`` for large exports (bigger raster
pixels). ``--world-map-coastline-width`` thickens Cartopy coastlines. For **smoother** raster
fields use ``--map_interpolation bilinear`` or ``bicubic`` (or finer ``--map_lon_bins`` /
``--map_lat_bins``). Empty raster bins can be filled from neighbors with ``--map-fill-empty nearest`` (default;
requires scipy); by default empty **ocean** bins are not inpainted (use ``--map-fill-ocean`` for that;
land mask needs cartopy + shapely). ``--world_map_point_size`` sets geographic scatter marker **diameter**
(converted to matplotlib ``s``); add ``--world-map-scatter-raw-s`` for raw ``s``. PC1–PC2 uses
``--scatter_point_size``.

Example:
  cd explainable-earth-embeddings
  python scripts/satclip_location_pca_map.py \\
    --index_csv /home/seri6958/data/satclip_s2_100k/index.csv \\
    --output_dir ./satclip_pca_maps

  Optional subsample for speed: ``--max_samples 50000`` (default uses all rows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_CLIP_SURGERY = _REPO_ROOT / "CLIP_Surgery"
if str(_CLIP_SURGERY) not in sys.path:
    sys.path.insert(0, str(_CLIP_SURGERY))

from images.clip_surgery import get_satclip  # noqa: E402

DEFAULT_MODEL = "microsoft/SatCLIP-ViT16-L40"
DEFAULT_CKPT = "satclip-vit16-l40.ckpt"

# Matplotlib colormap for per-PC world maps (raster + geographic scatter), PC1×PC2 vs latitude scatter,
# and PC1–PC3 RGB composite (``direct`` mode: three colors sampled from this map, weighted by PCs).
WORLD_MAP_PC_CMAP: str = "summer"

# Alpha for per-sample scatter when drawn on top of raster world maps (see --map-overlay-points).
MAP_OVERLAY_SCATTER_ALPHA: float = 0.5

# Reference cities (name, latitude, longitude) for red markers on PC1–PC2 and world maps.
HIGHLIGHT_CITIES: tuple[tuple[str, float, float], ...] = (
    ("Paris", 48.86, 2.3),
    ("New York", 40.7128, -74.0060),
)


def load_lon_lat_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return lon, lat arrays from S2-100K style (fn,lon,lat) or Places style (id,lat,lon)."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "lon" in cols and "lat" in cols:
        lon = df[cols["lon"]].to_numpy(dtype=np.float64)
        lat = df[cols["lat"]].to_numpy(dtype=np.float64)
    elif "longitude" in cols and "latitude" in cols:
        lon = df[cols["longitude"]].to_numpy(dtype=np.float64)
        lat = df[cols["latitude"]].to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Expected columns lon/lat (or longitude/latitude), got {list(df.columns)}"
        )
    return lon, lat


@torch.no_grad()
def encode_locations_batched(
    model: torch.nn.Module,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    device: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """Encode (lon, lat) with SatCLIP location encoder; returns (N, D) float32."""
    n = lon.shape[0]
    encode = model.encode_location
    out_list: list[np.ndarray] = []
    for start in tqdm(range(0, n, batch_size), desc="location encoder"):
        end = min(start + batch_size, n)
        t = torch.stack(
            [
                torch.tensor(lon[start:end], device=device, dtype=torch.float64),
                torch.tensor(lat[start:end], device=device, dtype=torch.float64),
            ],
            dim=1,
        )
        emb = encode(t).float()
        if emb.dim() == 3 and emb.shape[1] == 1:
            emb = emb.squeeze(1)
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        out_list.append(emb.cpu().numpy())
    return np.concatenate(out_list, axis=0)


def encode_highlight_city_scores(
    model: torch.nn.Module,
    cities: tuple[tuple[str, float, float], ...],
    *,
    device: str,
    normalize: bool,
    scaler: StandardScaler | None,
    pca: PCA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Encode each city with the location encoder and project with the same PCA as the index."""
    if not cities:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.empty((0, pca.n_components_)),
            [],
        )
    names = [c[0] for c in cities]
    lats = np.array([c[1] for c in cities], dtype=np.float64)
    lons = np.array([c[2] for c in cities], dtype=np.float64)
    emb = encode_locations_batched(
        model,
        lons,
        lats,
        device=device,
        batch_size=max(len(cities), 1),
        normalize=normalize,
    )
    row = np.asarray(emb, dtype=np.float64)
    if scaler is not None:
        row = scaler.transform(row)
    city_scores = pca.transform(row)
    return lons, lats, city_scores, names


def _annotate_cities_geo(
    ax,
    *,
    use_cartopy: bool,
    city_lons: np.ndarray | None,
    city_lats: np.ndarray | None,
    city_names: list[str] | None,
) -> None:
    """Red markers and city names on equirectangular maps (on top of other layers)."""
    if (
        city_lons is None
        or city_lats is None
        or city_names is None
        or len(city_lons) == 0
    ):
        return
    if use_cartopy:
        import cartopy.crs as ccrs

        tr = ccrs.PlateCarree()
        for i in range(len(city_lons)):
            ax.plot(
                city_lons[i],
                city_lats[i],
                marker="o",
                color="red",
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.9,
                transform=tr,
                zorder=250,
                linestyle="none",
            )
            ax.text(
                city_lons[i],
                city_lats[i],
                f"  {city_names[i]}",
                transform=tr,
                fontsize=11,
                fontweight="bold",
                color="red",
                zorder=251,
                ha="left",
                va="bottom",
                clip_on=False,
            )
    else:
        for i in range(len(city_lons)):
            ax.plot(
                city_lons[i],
                city_lats[i],
                marker="o",
                color="red",
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linestyle="none",
                zorder=250,
            )
            ax.annotate(
                city_names[i],
                (city_lons[i], city_lats[i]),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="red",
                zorder=251,
            )


def _hist2d_mean(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    n_lon: int,
    n_lat: int,
) -> np.ndarray:
    """Mean ``values`` per lon/lat bin. Shape ``(n_lat, n_lon)`` for ``imshow(..., origin='lower')``."""
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    sum_w, _, _ = np.histogram2d(
        lon,
        lat,
        bins=[lon_edges, lat_edges],
        weights=values.astype(np.float64),
    )
    cnt, _, _ = np.histogram2d(lon, lat, bins=[lon_edges, lat_edges])
    with np.errstate(invalid="ignore"):
        grid = sum_w / cnt
    grid[cnt == 0] = np.nan
    # H[i,j] = lon-bin i, lat-bin j → transpose so rows = latitude
    return grid.T


_LAND_MASK_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _land_mask_grid(n_lon: int, n_lat: int) -> np.ndarray:
    """Return bool array (n_lat, n_lon): True if cell center lies on Natural Earth land."""
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        from shapely.vectorized import contains
    except ImportError as e:
        raise SystemExit(
            "Install cartopy and shapely to keep ocean holes unfilled with "
            "--map-fill-empty nearest (or pass --map-fill-ocean to fill all cells). "
            "Example: pip install cartopy shapely"
        ) from e
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon2d, lat2d = np.meshgrid(lon_c, lat_c)
    land_shp = shpreader.natural_earth(
        resolution="50m", category="physical", name="land"
    )
    reader = shpreader.Reader(land_shp)
    land_union = unary_union(list(reader.geometries()))
    return np.asarray(contains(land_union, lon2d, lat2d), dtype=bool)


def _get_land_mask_grid(n_lon: int, n_lat: int) -> np.ndarray:
    key = (n_lon, n_lat)
    if key not in _LAND_MASK_CACHE:
        _LAND_MASK_CACHE[key] = _land_mask_grid(n_lon, n_lat)
    return _LAND_MASK_CACHE[key]


def _fill_nan_nearest_2d(
    grid: np.ndarray,
    *,
    land_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Replace NaN with the value at the nearest finite cell (Euclidean distance in bin space).

    If ``land_mask`` is provided (same shape as ``grid``), any bin that was NaN and lies
    over ocean (``land_mask`` False) is left NaN so empty ocean is not inpainted.
    """
    grid = np.asarray(grid, dtype=np.float64)
    orig_nan = np.isnan(grid)
    if not np.any(orig_nan):
        return grid
    if not np.any(np.isfinite(grid)):
        return grid
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as e:
        raise SystemExit(
            "Install scipy to use --map-fill-empty nearest (e.g. pip install scipy)"
        ) from e
    # 0 = source (finite), non-zero = need distance to nearest source
    m = np.where(np.isfinite(grid), 0, 1)
    _, indices = distance_transform_edt(m, return_distances=True, return_indices=True)
    filled = grid[indices[0], indices[1]]
    out = np.where(orig_nan, filled, grid)
    if land_mask is not None:
        if land_mask.shape != grid.shape:
            raise ValueError("land_mask shape must match grid")
        ocean_empty = orig_nan & (~land_mask)
        out = np.where(ocean_empty, np.nan, out)
    return out


def _world_map_matplotlib_scatter_s(
    world_map_point_size: float,
    *,
    raw_area: bool,
) -> float:
    """Matplotlib ``scatter(..., s=…)`` is marker *area* in points².

    By default we treat ``world_map_point_size`` as approximate **diameter in points**
    (like ``plot`` marker size), which makes changes much more visible than raw area.
    With ``raw_area=True``, the value is passed through as matplotlib ``s`` (legacy).
    """
    x = max(float(world_map_point_size), 0.25)
    if raw_area:
        return x
    r = x / 2.0
    return float(np.pi * r * r)


def _cartopy_available() -> bool:
    try:
        import cartopy  # noqa: F401

        return True
    except ImportError:
        return False


def _style_world_map_axes(ax) -> None:
    """No grid, no tick labels, no spines (map-only frame)."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_axis_off()


def _cartopy_base_layers(ax, cfeature, *, z: float = 0) -> None:
    """Land fill + ocean under data; Natural Earth 50m."""
    ax.add_feature(
        cfeature.LAND.with_scale("50m"),
        facecolor="0.94",
        edgecolor="none",
        zorder=z,
    )
    ax.add_feature(cfeature.OCEAN, facecolor="0.88", zorder=z)


def _cartopy_coastlines(ax, cfeature, *, z: float, lw: float = 1.0) -> None:
    ax.add_feature(
        cfeature.COASTLINE.with_scale("50m"),
        linewidth=lw,
        edgecolor="0.25",
        zorder=z,
    )


def plot_pc_world_map(
    lon: np.ndarray,
    lat: np.ndarray,
    z: np.ndarray,
    title: str,
    out_path: Path,
    *,
    use_cartopy: bool,
    raster: bool,
    n_lon: int,
    n_lat: int,
    interpolation: str,
    world_map_point_size: float,
    world_map_figsize: tuple[float, float],
    world_map_dpi: int,
    world_map_scatter_raw_s: bool,
    coastline_width: float,
    map_fill_empty: str,
    map_fill_ocean: bool,
    map_overlay_points: bool,
    city_lons: np.ndarray | None = None,
    city_lats: np.ndarray | None = None,
    city_names: list[str] | None = None,
) -> None:
    fig = plt.figure(figsize=world_map_figsize)
    s_markers = _world_map_matplotlib_scatter_s(
        world_map_point_size, raw_area=world_map_scatter_raw_s
    )
    if raster:
        grid = _hist2d_mean(lon, lat, z, n_lon, n_lat)
        if map_fill_empty == "nearest":
            lm = None if map_fill_ocean else _get_land_mask_grid(n_lon, n_lat)
            grid = _fill_nan_nearest_2d(grid, land_mask=lm)
        if not np.any(np.isfinite(grid)):
            raise ValueError("Raster map has no finite cells; check lon/lat range.")
        masked = np.ma.masked_invalid(grid)
        cmap = plt.get_cmap(WORLD_MAP_PC_CMAP).copy()
        cmap.set_bad(color=(0.85, 0.85, 0.85, 1.0))
        vmin = float(np.nanmin(grid))
        vmax = float(np.nanmax(grid))
        if use_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_global()
            _cartopy_base_layers(ax, cfeature, z=0)
            im = ax.imshow(
                masked,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(-180, 180, -90, 90),
                origin="lower",
                transform=ccrs.PlateCarree(),
                interpolation=interpolation,
                aspect="auto",
                zorder=1,
            )
            if map_overlay_points:
                ax.scatter(
                    lon,
                    lat,
                    c=z,
                    s=s_markers,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=MAP_OVERLAY_SCATTER_ALPHA,
                    transform=ccrs.PlateCarree(),
                    edgecolors="none",
                    linewidths=0,
                    rasterized=False,
                    zorder=2,
                )
            _cartopy_coastlines(
                ax, cfeature, z=3 if map_overlay_points else 2, lw=coastline_width
            )
            plt.colorbar(im, ax=ax, shrink=0.5, label="score")
            _style_world_map_axes(ax)
        else:
            ax = fig.add_subplot(1, 1, 1)
            im = ax.imshow(
                masked,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                extent=(-180, 180, -90, 90),
                origin="lower",
                interpolation=interpolation,
                aspect="equal",
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_facecolor("0.92")
            if map_overlay_points:
                ax.scatter(
                    lon,
                    lat,
                    c=z,
                    s=s_markers,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=MAP_OVERLAY_SCATTER_ALPHA,
                    edgecolors="none",
                    linewidths=0,
                    rasterized=False,
                    zorder=2,
                )
            plt.colorbar(im, ax=ax, shrink=0.6, label="score")
            _style_world_map_axes(ax)
        if map_fill_empty == "nearest":
            if map_fill_ocean:
                empty_note = "empty=nearest fill (incl. ocean)"
            else:
                empty_note = "empty=nearest fill (land only)"
        else:
            empty_note = "empty=masked"
        ov = " + points" if map_overlay_points else ""
        sub = f"mean per cell · grid {n_lon}×{n_lat} · {empty_note}{ov}"
    elif use_cartopy:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()
        _cartopy_base_layers(ax, cfeature, z=0)
        sc = ax.scatter(
            lon,
            lat,
            c=z,
            s=s_markers,
            cmap=WORLD_MAP_PC_CMAP,
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            edgecolors="none",
            linewidths=0,
            rasterized=False,
            zorder=2,
        )
        _cartopy_coastlines(ax, cfeature, z=3, lw=coastline_width)
        plt.colorbar(sc, ax=ax, shrink=0.5, label="score")
        _style_world_map_axes(ax)
        sub = "scatter"
    else:
        ax = fig.add_subplot(1, 1, 1)
        sc = ax.scatter(
            lon,
            lat,
            c=z,
            s=s_markers,
            cmap=WORLD_MAP_PC_CMAP,
            alpha=0.85,
            edgecolors="none",
            linewidths=0,
            rasterized=False,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.set_facecolor("0.9")
        plt.colorbar(sc, ax=ax, shrink=0.6, label="score")
        _style_world_map_axes(ax)
        sub = "scatter"

    _annotate_cities_geo(
        ax,
        use_cartopy=use_cartopy,
        city_lons=city_lons,
        city_lats=city_lats,
        city_names=city_names,
    )
    ax.set_title(title + ("\n" + sub if raster else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=world_map_dpi, bbox_inches="tight")
    plt.close(fig)


def _normalize_pc_columns_01(scores_3: np.ndarray) -> np.ndarray:
    """Min–max each PC column to [0, 1] (same shape)."""
    out = np.zeros_like(scores_3, dtype=np.float64)
    for j in range(scores_3.shape[1]):
        col = scores_3[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi - lo < 1e-12:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - lo) / (hi - lo)
    return out


def _pc123_to_blue_yellow_purple_rgb(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """Blend PC intensities: PC1→blue, PC2→yellow, PC3→purple (additive RGB, clipped)."""
    # Blue (0,0,v1), Yellow (v2,v2,0), Purple (v3,0,v3) → R=v2+v3, G=v2, B=v1+v3
    R = np.clip(v2 + v3, 0.0, 1.0)
    G = np.clip(v2, 0.0, 1.0)
    B = np.clip(v1 + v3, 0.0, 1.0)
    return np.stack([R, G, B], axis=-1).astype(np.float32)


def _three_basis_colors_from_cmap(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pick three distinct RGB basis vectors from a matplotlib colormap (for PC1–PC3 weighting)."""
    cmap = plt.get_cmap(name)
    n = int(getattr(cmap, "N", 256))
    if n <= 3:
        t0, t1, t2 = 0.0, 0.5, 1.0
    elif n <= 32:
        i0, i1, i2 = 0, n // 2, n - 1
        t0 = (i0 + 0.5) / n
        t1 = (i1 + 0.5) / n
        t2 = (i2 + 0.5) / n
    else:
        t0, t1, t2 = 0.2, 0.5, 0.8
    return tuple(
        np.asarray(cmap(t)[:3], dtype=np.float64) for t in (t0, t1, t2)
    )


def _pc123_to_direct_rgb(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """Direct PC-to-RGB mapping: R=PC1, G=PC2, B=PC3 (each in [0, 1]), clipped."""
    v1 = np.clip(np.asarray(v1, dtype=np.float64), 0.0, 1.0)
    v2 = np.clip(np.asarray(v2, dtype=np.float64), 0.0, 1.0)
    v3 = np.clip(np.asarray(v3, dtype=np.float64), 0.0, 1.0)
    rgb = np.stack([v1, v2, v3], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _rgb01_to_u8(rgb01: np.ndarray) -> np.ndarray:
    """Convert RGB float array in [0, 1] to clipped uint8 [0, 255]."""
    rgb01 = np.asarray(rgb01, dtype=np.float64)
    return np.clip(np.rint(rgb01 * 255.0), 0.0, 255.0).astype(np.uint8)


def _pc123_normalized_to_rgb(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    if mode == "direct":
        return _pc123_to_direct_rgb(v1, v2, v3)
    if mode == "blend":
        return _pc123_to_blue_yellow_purple_rgb(v1, v2, v3)
    raise ValueError(f"Unknown pc123 color mode: {mode!r}")


def _pc123_rgb_subtitle(
    mode: str,
    *,
    raster: bool,
    n_lon: int | None = None,
    n_lat: int | None = None,
    map_fill_empty: str = "none",
    map_fill_ocean: bool = False,
    map_overlay_points: bool = False,
) -> str:
    if mode == "direct":
        enc = "R=PC1, G=PC2, B=PC3"
    else:
        enc = "PC1=blue, PC2=yellow, PC3=purple (additive blend)"
    if raster and n_lon is not None and n_lat is not None:
        if map_fill_empty == "nearest":
            if map_fill_ocean:
                empty_note = "empty=nearest fill (incl. ocean)"
            else:
                empty_note = "empty=nearest fill (land only)"
        else:
            empty_note = "empty=gray"
        ov = " + points" if map_overlay_points else ""
        return f"{enc} · mean per cell · grid {n_lon}×{n_lat} · {empty_note}{ov}"
    return f"{enc} (min–max per PC)"


def _pc123_scores_to_assigned_rgb(scores_3: np.ndarray, *, mode: str) -> np.ndarray:
    """Per-column min–max, then encode as clipped uint8 RGB per ``mode`` (N×3)."""
    assert scores_3.shape[1] == 3
    v = _normalize_pc_columns_01(scores_3)
    rgb01 = _pc123_normalized_to_rgb(v[:, 0], v[:, 1], v[:, 2], mode=mode)
    return _rgb01_to_u8(rgb01)


def _pc123_grids_to_rgb01(
    lon: np.ndarray,
    lat: np.ndarray,
    scores_pc123: np.ndarray,
    n_lon: int,
    n_lat: int,
    *,
    mode: str,
    map_fill_empty: str,
    map_fill_ocean: bool,
) -> np.ndarray:
    """Mean PC1–PC3 per cell; min–max each PC grid; map to RGB per ``mode``; empty → gray."""
    assert scores_pc123.shape[1] == 3
    g0 = _hist2d_mean(lon, lat, scores_pc123[:, 0], n_lon, n_lat)
    g1 = _hist2d_mean(lon, lat, scores_pc123[:, 1], n_lon, n_lat)
    g2 = _hist2d_mean(lon, lat, scores_pc123[:, 2], n_lon, n_lat)
    if map_fill_empty == "nearest":
        lm = None if map_fill_ocean else _get_land_mask_grid(n_lon, n_lat)
        g0 = _fill_nan_nearest_2d(g0, land_mask=lm)
        g1 = _fill_nan_nearest_2d(g1, land_mask=lm)
        g2 = _fill_nan_nearest_2d(g2, land_mask=lm)
    stacked = np.stack([g0, g1, g2], axis=-1)
    v = np.full(stacked.shape, np.nan, dtype=np.float64)
    for j in range(3):
        ch = stacked[..., j]
        valid = np.isfinite(ch)
        if not np.any(valid):
            continue
        lo, hi = float(np.nanmin(ch)), float(np.nanmax(ch))
        if hi - lo < 1e-12:
            v[..., j] = np.where(valid, 0.5, np.nan)
        else:
            v[..., j] = np.where(valid, (ch - lo) / (hi - lo), np.nan)
    valid = np.isfinite(v).all(axis=-1)
    v1 = np.where(valid, v[..., 0], 0.0)
    v2 = np.where(valid, v[..., 1], 0.0)
    v3 = np.where(valid, v[..., 2], 0.0)
    rgb = _pc123_normalized_to_rgb(v1, v2, v3, mode=mode)
    gray = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    rgb = np.where(valid[..., None], rgb, gray)
    return _rgb01_to_u8(np.clip(rgb, 0.0, 1.0))


def plot_pc123_rgb_world_map(
    lon: np.ndarray,
    lat: np.ndarray,
    scores_pc123: np.ndarray,
    title: str,
    out_path: Path,
    *,
    use_cartopy: bool,
    explained_var_ratios: np.ndarray | None = None,
    raster: bool,
    n_lon: int,
    n_lat: int,
    interpolation: str,
    world_map_point_size: float,
    world_map_figsize: tuple[float, float],
    world_map_dpi: int,
    world_map_scatter_raw_s: bool,
    pc123_color_mode: str,
    coastline_width: float,
    map_fill_empty: str,
    map_fill_ocean: bool,
    map_overlay_points: bool,
    city_lons: np.ndarray | None = None,
    city_lats: np.ndarray | None = None,
    city_names: list[str] | None = None,
) -> None:
    """PC1–PC3 as RGB (scatter or raster mean per lon/lat cell); see ``pc123_color_mode``."""
    assert scores_pc123.shape[1] == 3
    fig = plt.figure(figsize=world_map_figsize)
    s_markers = _world_map_matplotlib_scatter_s(
        world_map_point_size, raw_area=world_map_scatter_raw_s
    )
    if raster:
        rgb = _pc123_grids_to_rgb01(
            lon,
            lat,
            scores_pc123,
            n_lon,
            n_lat,
            mode=pc123_color_mode,
            map_fill_empty=map_fill_empty,
            map_fill_ocean=map_fill_ocean,
        )
        rgb_pts = (
            _pc123_scores_to_assigned_rgb(scores_pc123, mode=pc123_color_mode)
            if map_overlay_points
            else None
        )
        rgb_pts_scatter = (
            rgb_pts.astype(np.float32) / 255.0 if rgb_pts is not None else None
        )
        if use_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_global()
            _cartopy_base_layers(ax, cfeature, z=0)
            ax.imshow(
                rgb,
                extent=(-180, 180, -90, 90),
                origin="lower",
                transform=ccrs.PlateCarree(),
                interpolation=interpolation,
                aspect="auto",
                zorder=1,
            )
            if rgb_pts_scatter is not None:
                ax.scatter(
                    lon,
                    lat,
                    c=rgb_pts_scatter,
                    s=s_markers,
                    alpha=MAP_OVERLAY_SCATTER_ALPHA,
                    transform=ccrs.PlateCarree(),
                    edgecolors="none",
                    linewidths=0,
                    rasterized=False,
                    zorder=2,
                )
            _cartopy_coastlines(
                ax, cfeature, z=3 if map_overlay_points else 2, lw=coastline_width
            )
            _style_world_map_axes(ax)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(
                rgb,
                extent=(-180, 180, -90, 90),
                origin="lower",
                interpolation=interpolation,
                aspect="equal",
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_facecolor("0.92")
            if rgb_pts_scatter is not None:
                ax.scatter(
                    lon,
                    lat,
                    c=rgb_pts_scatter,
                    s=s_markers,
                    alpha=MAP_OVERLAY_SCATTER_ALPHA,
                    edgecolors="none",
                    linewidths=0,
                    rasterized=False,
                    zorder=2,
                )
            _style_world_map_axes(ax)
        sub = _pc123_rgb_subtitle(
            pc123_color_mode,
            raster=True,
            n_lon=n_lon,
            n_lat=n_lat,
            map_fill_empty=map_fill_empty,
            map_fill_ocean=map_fill_ocean,
            map_overlay_points=map_overlay_points,
        )
    else:
        rgb_pts = _pc123_scores_to_assigned_rgb(scores_pc123, mode=pc123_color_mode)
        rgb_pts_scatter = rgb_pts.astype(np.float32) / 255.0
        if use_cartopy:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_global()
            _cartopy_base_layers(ax, cfeature, z=0)
            ax.scatter(
                lon,
                lat,
                c=rgb_pts_scatter,
                s=s_markers,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                edgecolors="none",
                linewidths=0,
                rasterized=False,
                zorder=2,
            )
            _cartopy_coastlines(ax, cfeature, z=3, lw=coastline_width)
            _style_world_map_axes(ax)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.scatter(
                lon,
                lat,
                c=rgb_pts_scatter,
                s=s_markers,
                alpha=0.9,
                edgecolors="none",
                linewidths=0,
                rasterized=False,
            )
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")
            ax.set_facecolor("0.9")
            _style_world_map_axes(ax)
        sub = _pc123_rgb_subtitle(pc123_color_mode, raster=False)

    if explained_var_ratios is not None and len(explained_var_ratios) >= 3:
        sub += (
            f" · var PC1–3: {explained_var_ratios[0] * 100:.1f}% / "
            f"{explained_var_ratios[1] * 100:.1f}% / {explained_var_ratios[2] * 100:.1f}%"
        )
    _annotate_cities_geo(
        ax,
        use_cartopy=use_cartopy,
        city_lons=city_lons,
        city_lats=city_lats,
        city_names=city_names,
    )
    ax.set_title(title + "\n" + sub, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=world_map_dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_world_map_figsize(s: str) -> tuple[float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'W,H' inches, got {s!r}")
    return float(parts[0]), float(parts[1])


def main() -> None:
    # Copy-paste from docs often includes literal "..." — strip it so argparse does not fail.
    sys.argv = [a for a in sys.argv if a != "..."]

    p = argparse.ArgumentParser(
        description="PCA of SatCLIP location embeddings + world maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/satclip_location_pca_map.py \\\n"
            "    --index_csv ~/data/satclip_s2_100k/index.csv \\\n"
            "    --no-map-raster --world_map_point_size 15\n"
            "\n"
            "Do not pass the three-dot placeholder as an argument; it is ignored if present."
        ),
    )
    p.add_argument(
        "--index_csv",
        type=Path,
        default=Path.home() / "data/satclip_s2_100k/index.csv",
        help="CSV with lon/lat (S2-100K: fn,lon,lat) or id,lat,lon",
    )
    p.add_argument("--output_dir", type=Path, default=Path("satclip_location_pca_out"))
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Max rows from the index to use (0 = all rows; default: 0)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--n_components", type=int, default=16)
    p.add_argument(
        "--plot_components",
        type=int,
        default=6,
        help="Number of leading PCs to draw as world maps",
    )
    p.add_argument(
        "--normalize_embeddings",
        action="store_true",
        help="L2-normalize each embedding (as in contrastive training)",
    )
    p.add_argument(
        "--standardize_before_pca",
        action="store_true",
        help="Apply sklearn StandardScaler before PCA (per-dimension z-score)",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--hf_model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face repo id for SatCLIP checkpoint",
    )
    p.add_argument("--hf_file", type=str, default=DEFAULT_CKPT)
    p.add_argument(
        "--surgery",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SatCLIPSurgeryLightningModule (default: --surgery). "
        "Use --no-surgery for the standard SatCLIP checkpoint loader.",
    )
    p.add_argument(
        "--cartopy",
        action="store_true",
        help="Require Cartopy for world maps (error if not installed). "
        "If omitted, land/ocean/coastlines are still drawn when Cartopy is available "
        "(see --world-map-continents).",
    )
    p.add_argument(
        "--world-map-continents",
        dest="world_map_continents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw continents (Cartopy Natural Earth land, ocean, coastlines) when installed "
        "(default: on). Use --no-world-map-continents for a bare lon/lat plot without geography.",
    )
    p.add_argument(
        "--superpose-pc-rgb",
        dest="superpose_pc_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write world_map_pc123_rgb.png (default: on); encoding is set by --pc123-color-mode",
    )
    p.add_argument(
        "--pc123-color-mode",
        choices=["direct", "blend"],
        default="direct",
        help="RGB for world_map_pc123_rgb: direct = three colors from WORLD_MAP_PC_CMAP "
        "(module global) weighted by PC1–PC3 after min–max each PC (default); "
        "blend = older blue/yellow/purple additive mix",
    )
    p.add_argument(
        "--map-raster",
        dest="map_raster",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Paint world maps as lon/lat grid cells (mean per cell); "
        "use --no-map-raster for point scatter only (default: raster)",
    )
    p.add_argument(
        "--map-overlay-points",
        dest="map_overlay_points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With raster world maps, draw each sample on top (semi-transparent; default: on). "
        "Use --no-map-overlay-points for filled raster only.",
    )
    p.add_argument(
        "--map_lon_bins",
        type=int,
        default=360,
        help="Number of longitude bins for raster maps (default: 360, 1°)",
    )
    p.add_argument(
        "--map_lat_bins",
        type=int,
        default=180,
        help="Number of latitude bins for raster maps (default: 180, 1°)",
    )
    p.add_argument(
        "--map_interpolation",
        choices=["none", "nearest", "bilinear", "bicubic", "antialiased"],
        default="nearest",
        help="Raster maps only: matplotlib imshow interpolation between lon/lat cells. "
        "nearest = sharp block edges (default); bilinear / bicubic = smoother blending "
        "(softer boundaries, some blur); antialiased = resampling with smoothing. "
        "none is treated as nearest.",
    )
    p.add_argument(
        "--map-fill-empty",
        dest="map_fill_empty",
        choices=["none", "nearest"],
        default="nearest",
        help="Raster maps only: fill lon/lat bins with no samples. "
        "nearest = copy value from nearest non-empty cell (needs scipy; default). "
        "Empty ocean bins stay unfilled unless --map-fill-ocean. "
        "none = leave holes masked / gray in PC maps.",
    )
    p.add_argument(
        "--map-fill-ocean",
        dest="map_fill_ocean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="With --map-fill-empty nearest, also inpaint empty ocean bins (default: off). "
        "When off, Natural Earth land mask is used (needs cartopy + shapely).",
    )
    p.add_argument(
        "--scatter_point_size",
        type=float,
        default=12.0,
        help="Marker size for PC1–PC2 scatter (matplotlib ``s``, points^2 area)",
    )
    p.add_argument(
        "--world_map_point_size",
        type=float,
        default=8.0,
        help="World-map scatter marker scale: by default this is approximate **diameter in "
        "points** (then converted to matplotlib ``s`` = area); try 4–20. "
        "Use --world-map-scatter-raw-s for legacy behavior (value = raw ``s`` area).",
    )
    p.add_argument(
        "--world-map-scatter-raw-s",
        dest="world_map_scatter_raw_s",
        action="store_true",
        help="Treat --world_map_point_size as matplotlib scatter ``s`` (area in points²) "
        "without diameter conversion (matches old scripts).",
    )
    p.add_argument(
        "--world_map_figsize",
        type=str,
        default="20,10",
        help="World-map figure size in inches as W,H (default: 20,10; larger = bigger raster cells)",
    )
    p.add_argument(
        "--world_map_dpi",
        type=int,
        default=200,
        help="PNG resolution for world_map_*.png (default: 200)",
    )
    p.add_argument(
        "--world-map-coastline-width",
        dest="world_map_coastline_width",
        type=float,
        default=1.0,
        help="Cartopy coastline stroke width in points (default: 1.0). Ignored without Cartopy.",
    )
    args = p.parse_args()

    if not args.index_csv.is_file():
        raise FileNotFoundError(f"Index CSV not found: {args.index_csv}")

    world_map_figsize = _parse_world_map_figsize(args.world_map_figsize)

    if args.cartopy:
        try:
            import cartopy  # noqa: F401
        except ImportError as e:
            raise SystemExit("Install cartopy or run without --cartopy") from e

    use_cartopy = args.cartopy or (
        args.world_map_continents and _cartopy_available()
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    lon, lat = load_lon_lat_csv(args.index_csv)
    n = lon.shape[0]
    if args.max_samples and args.max_samples > 0 and n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        lon, lat = lon[idx], lat[idx]
        n = args.max_samples

    ckpt = hf_hub_download(args.hf_model, args.hf_file)
    model = get_satclip(ckpt, device=device, surgery=args.surgery, return_all=True)
    model.eval()

    embeddings = encode_locations_batched(
        model,
        lon,
        lat,
        device=device,
        batch_size=args.batch_size,
        normalize=args.normalize_embeddings,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "coords_subset.npz",
        lon=lon,
        lat=lat,
        n_original=int(pd.read_csv(args.index_csv).shape[0]),
    )
    filename = "embeddings_surgery.npy" if args.surgery else "embeddings.npy"
    np.save(args.output_dir / filename, embeddings)

    x = embeddings
    scaler: StandardScaler | None = None
    if args.standardize_before_pca:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    n_comp = min(args.n_components, x.shape[1], x.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=args.seed)
    scores = pca.fit_transform(x)

    city_lons, city_lats, city_scores, city_names = encode_highlight_city_scores(
        model,
        HIGHLIGHT_CITIES,
        device=device,
        normalize=args.normalize_embeddings,
        scaler=scaler,
        pca=pca,
    )
    np.savez(
        args.output_dir / "highlight_cities_pca.npz",
        names=np.array(city_names, dtype=object),
        lon=city_lons,
        lat=city_lats,
        scores=city_scores,
    )

    np.save(args.output_dir / "pca_scores.npy", scores)
    np.save(args.output_dir / "pca_components.npy", pca.components_)
    np.save(args.output_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    # Scree / variance
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        np.arange(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        color="steelblue",
    )
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("SatCLIP location embedding PCA")
    fig.tight_layout()
    fig.savefig(args.output_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    n_plot = min(args.plot_components, scores.shape[1])
    map_interp = args.map_interpolation
    if map_interp == "none":
        map_interp = "nearest"
    for k in range(n_plot):
        plot_pc_world_map(
            lon,
            lat,
            scores[:, k],
            title=f"PC{k + 1} ({pca.explained_variance_ratio_[k] * 100:.2f}% variance)",
            out_path=args.output_dir / f"world_map_pc{k + 1:02d}.png",
            use_cartopy=use_cartopy,
            raster=args.map_raster,
            n_lon=args.map_lon_bins,
            n_lat=args.map_lat_bins,
            interpolation=map_interp,
            world_map_point_size=args.world_map_point_size,
            world_map_figsize=world_map_figsize,
            world_map_dpi=args.world_map_dpi,
            world_map_scatter_raw_s=args.world_map_scatter_raw_s,
            coastline_width=args.world_map_coastline_width,
            map_fill_empty=args.map_fill_empty,
            map_fill_ocean=args.map_fill_ocean,
            map_overlay_points=args.map_overlay_points and args.map_raster,
            city_lons=city_lons,
            city_lats=city_lats,
            city_names=city_names,
        )

    if args.superpose_pc_rgb and scores.shape[1] >= 3:
        plot_pc123_rgb_world_map(
            lon,
            lat,
            scores[:, :3],
            title=f"PC1–PC3 composite ({WORLD_MAP_PC_CMAP})",
            out_path=args.output_dir / "world_map_pc123_rgb.png",
            use_cartopy=use_cartopy,
            explained_var_ratios=pca.explained_variance_ratio_,
            raster=args.map_raster,
            n_lon=args.map_lon_bins,
            n_lat=args.map_lat_bins,
            interpolation=map_interp,
            world_map_point_size=args.world_map_point_size,
            world_map_figsize=world_map_figsize,
            world_map_dpi=args.world_map_dpi,
            world_map_scatter_raw_s=args.world_map_scatter_raw_s,
            pc123_color_mode=args.pc123_color_mode,
            coastline_width=args.world_map_coastline_width,
            map_fill_empty=args.map_fill_empty,
            map_fill_ocean=args.map_fill_ocean,
            map_overlay_points=args.map_overlay_points and args.map_raster,
            city_lons=city_lons,
            city_lats=city_lats,
            city_names=city_names,
        )

    # PC1 vs PC2 scatter (not geographic)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        scores[:, 0],
        scores[:, 1],
        c=lat,
        cmap=WORLD_MAP_PC_CMAP,
        s=args.scatter_point_size,
        alpha=0.6,
        rasterized=True,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(sc, ax=ax, label="Latitude")
    ax.set_title("PCA scores (colored by latitude)")
    if city_scores.shape[0] > 0 and city_scores.shape[1] >= 2:
        for i, name in enumerate(city_names):
            ax.scatter(
                city_scores[i, 0],
                city_scores[i, 1],
                c="red",
                s=max(args.scatter_point_size * 3, 36),
                zorder=10,
                edgecolors="white",
                linewidths=0.9,
            )
            ax.annotate(
                name,
                (city_scores[i, 0], city_scores[i, 1]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                color="red",
                zorder=11,
            )
    fig.tight_layout()
    fig.savefig(args.output_dir / "pc1_pc2_scatter_surgery.png", dpi=150)
    plt.close(fig)

    print(f"Wrote outputs under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

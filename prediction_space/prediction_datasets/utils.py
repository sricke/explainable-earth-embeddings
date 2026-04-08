"""Low-level helpers: path constants, CSV/shapefile loading, embedding cache."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype

_PREDICTION_SPACE = Path(__file__).resolve().parent.parent
if str(_PREDICTION_SPACE) not in sys.path:
    sys.path.insert(0, str(_PREDICTION_SPACE))
_REPO_ROOT = _PREDICTION_SPACE.parent
_satclip_ext = _REPO_ROOT / "external" / "satclip"
if _satclip_ext.is_dir() and str(_satclip_ext) not in sys.path:
    sys.path.insert(0, str(_satclip_ext))

# from embeddings.generate_embeddings import (
#     generate_aef_embeddings,
#     generate_geoclip_embeddings,
#     generate_satclip_embeddings,
# )

DEFAULT_PREDICTION_TASKS_DIR = Path.home() / "data" / "prediction_tasks"
DEFAULT_LAT_LON_GRID_CSV = DEFAULT_PREDICTION_TASKS_DIR / "all_WORLD_100000_lat_lon.csv"


def lat_lon_to_tensor(lat: np.ndarray, lon: np.ndarray) -> torch.Tensor:
    """``(N,)`` lat/lon → float tensor ``(N, 2)`` with columns ``[lat, lon]``."""
    return torch.from_numpy(
        np.column_stack([np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)])
    ).float()


def train_test_row_indices(
    n: int, *, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproducible train/test index split for ``n`` rows."""
    if n < 2:
        raise ValueError(f"Need at least 2 rows, got {n}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_test = max(1, min(int(round(n * test_size)), n - 1))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def load_csv(
    dir: Path | str,
    *,
    value_col: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(lat, lon, value)`` from a CSV file.

    If *value_col* is ``None``, the last column that is not *lat_col* or *lon_col* is used.
    """
    dir = Path(dir)
    # if path contains csv file
    if any(dir.glob("*.csv")):
        csv_path = next(dir.glob("*.csv"))
    else:
        raise ValueError(f"No CSV file found in {dir}")
    df = pd.read_csv(csv_path)
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Expected columns {lat_col!r} and {lon_col!r}; got {list(df.columns)}")
    if value_col is None:
        candidates = [c for c in df.columns if c not in (lat_col, lon_col)]
        if not candidates:
            raise ValueError(f"No value column found in {list(df.columns)}")
        value_col = candidates[-1]
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    val = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
    return lat[mask], lon[mask], val[mask]


def load_shapefile(
    shp_dir: Path | str,
    *,
    value_col: str,
    lat_lon_csv: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(lat, lon, value)`` by spatial-joining a lat/lon grid against a shapefile."""
    lat_lon_path = Path(lat_lon_csv).expanduser() if lat_lon_csv is not None else DEFAULT_LAT_LON_GRID_CSV
    if not lat_lon_path.is_file():
        raise FileNotFoundError(lat_lon_path)

    ll = pd.read_csv(lat_lon_path)
    lat = pd.to_numeric(ll["lat"], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(ll["lon"], errors="coerce").to_numpy(dtype=np.float64)

    points = gpd.GeoDataFrame(
        index=pd.RangeIndex(len(lat)),
        geometry=gpd.points_from_xy(lon, lat),
        crs=4326,
    )
    if any(shp_dir.glob("*.shp")):
        shp_path = next(shp_dir.glob("*.shp"))
    else:
        raise ValueError(f"No shapefile found in {shp_dir}")
    polys = gpd.read_file(shp_path)
    if value_col not in polys.columns:
        raise ValueError(f"Column {value_col!r} not in shapefile; have {list(polys.columns)}")
    if polys.crs is None:
        polys = polys.set_crs(4326)
    polys = polys.to_crs(4326)

    joined = points.sjoin(polys[[value_col, "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")].reindex(points.index)
    val = joined[value_col]
    value = val.to_numpy(dtype=np.float64) if is_numeric_dtype(val) else val.to_numpy()

    mask = np.isfinite(lat) & np.isfinite(lon)
    return lat[mask], lon[mask], value[mask]


def embeddings_cache_path(task_name: str, backend: str) -> Path:
    return DEFAULT_PREDICTION_TASKS_DIR / task_name / f"embeddings_{backend.lower()}.pt"


def load_embeddings(
    latlons: torch.Tensor,
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    task_name: str | None = None,
) -> torch.Tensor:
    """Load or compute embeddings for ``latlons`` ``(N, 2)`` (``[lat, lon]`` per row)."""
    if latlons.ndim != 2 or latlons.shape[1] != 2:
        raise ValueError("latlons must have shape (N, 2)")
    n = latlons.shape[0]

    path: Path | None = None
    if task_name is not None:
        path = embeddings_cache_path(task_name, backend)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            cached = torch.load(path, weights_only=True)
            if cached.shape[0] == n:
                return cached

    b = backend.lower()
    if b == "satclip":
        emb = generate_satclip_embeddings(latlons, device=device)
    elif b == "geoclip":
        emb = generate_geoclip_embeddings(latlons, device=device)
    elif b == "aef":
        import ee
        ee.Authenticate()
        ee.Initialize(project="avid-poet-486323-a3")
        emb = generate_aef_embeddings(latlons)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'satclip', 'geoclip', or 'aef'")

    if path is not None:
        torch.save(emb, path)
    return emb

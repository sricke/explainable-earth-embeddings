"""Shared helpers for loading (lat, lon, value) from local prediction task files."""

from __future__ import annotations

import sys
from collections.abc import Callable
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

from embeddings.generate_embeddings import (
    generate_aef_embeddings,
    generate_geoclip_embeddings,
    generate_satclip_embeddings,
)

DEFAULT_PREDICTION_TASKS_DIR = Path.home() / "data" / "prediction_tasks"
DEFAULT_LAT_LON_GRID_CSV = DEFAULT_PREDICTION_TASKS_DIR / "all_WORLD_100000_lat_lon.csv"


def train_test_row_indices(
    n: int,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single random train/test index split into ``[0, n)`` (disjoint, full cover).

    Indices are sorted within each set. Uses ``numpy.random.RandomState`` for reproducibility.
    """
    if n < 2:
        raise ValueError(f"Need at least 2 rows for a train/test split, got {n}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    rng = np.random.RandomState(random_state)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_test = max(1, min(n_test, n - 1))
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


def load_shapefile_value_at_lat_lon_csv(
    shapefile_path_or_dir: Path | str,
    *,
    value_col: str,
    lat_lon_csv: Path | str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ``lat`` / ``lon`` from ``lat_lon_csv`` (default ``all_WORLD_100000_lat_lon.csv``),
    then set ``value_col`` from the polygon that contains each point (``predicate='within'``).

    Points outside all polygons get ``nan`` in ``value``. If a point matches multiple polygons,
    the first match is kept. Returns arrays aligned with the CSV rows (after dropping
    non-finite ``lat``/``lon``). Numeric ``value_col`` is returned as ``float64``; non-numeric
    columns (e.g. ISO codes) are returned as a NumPy array (typically ``object`` dtype).
    """
    lat_lon_path = Path(lat_lon_csv).expanduser() if lat_lon_csv is not None else DEFAULT_LAT_LON_GRID_CSV
    if not lat_lon_path.is_file():
        raise FileNotFoundError(lat_lon_path)

    ll = pd.read_csv(lat_lon_path)
    if lat_col not in ll.columns or lon_col not in ll.columns:
        raise ValueError(
            f"Expected columns {lat_col!r} and {lon_col!r}; got {list(ll.columns)}"
        )
    lat = pd.to_numeric(ll[lat_col], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(ll[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    n = lat.shape[0]
    points = gpd.GeoDataFrame(
        index=pd.RangeIndex(n),
        geometry=gpd.points_from_xy(lon, lat),
        crs=4326,
    )

    p = Path(shapefile_path_or_dir).expanduser()
    if p.is_dir():
        matches = sorted(p.glob("*.shp"))
        if not matches:
            raise FileNotFoundError(f"No .shp under {p}")
        if len(matches) > 1:
            raise ValueError(f"Multiple shapefiles in {p}; pass a specific .shp path: {matches}")
        shp = matches[0]
    else:
        shp = p

    polys = gpd.read_file(shp)
    if value_col not in polys.columns:
        raise ValueError(f"Column {value_col!r} not in shapefile; have {list(polys.columns)}")
    if polys.crs is None:
        polys = polys.set_crs(4326)
    polys = polys.to_crs(4326)

    joined = points.sjoin(
        polys[[value_col, "geometry"]], how="left", predicate="within"
    )
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.reindex(points.index)
    s = joined[value_col]
    if is_numeric_dtype(s):
        val = s.to_numpy(dtype=np.float64)
    else:
        val = s.to_numpy()

    mask = np.isfinite(lat) & np.isfinite(lon)
    return lat[mask], lon[mask], val[mask]


def _resolve_outcomes_csv_path(path_or_dir: Path | str, glob_pattern: str = "outcomes_sampled_*.csv") -> Path:
    # for data downloaded from code ocean
    # the data is in a directory with the name of the prediction task
    # data is a 
    p = Path(path_or_dir).expanduser()
    if p.is_dir():
        matches = sorted(p.glob(glob_pattern))
        if not matches:
            raise FileNotFoundError(
                f"No CSV matching {glob_pattern!r} under {p}. "
                "Place an outcomes CSV in that directory or pass a file path."
            )
        if len(matches) > 1:
            raise ValueError(f"Multiple CSVs in {p}; pass a specific file path: {matches}")
        return matches[0]
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def load_outcomes_sampled_csv(
    path_or_dir: Path | str,
    *,
    value_col: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    glob_pattern: str = "outcomes_sampled_*.csv",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load rows from an `outcomes_sampled_*.csv`-style file.
    Only works for data downloaded from code ocean (from MOSAIKS Repo)

    Returns
    -------
    lat, lon, value
        Each 1-D float array of length N.
    """
    csv_path = _resolve_outcomes_csv_path(path_or_dir, glob_pattern=glob_pattern)
    df = pd.read_csv(csv_path)

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"Expected columns {lat_col!r} and {lon_col!r}; got {list(df.columns)}"
        )
    if value_col is None:
        raise ValueError(f"value_col must be specified; got None")
    
    lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=np.float64)
    lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=np.float64)
    val = df[value_col]
    if val.dtype == object or val.dtype == str:
        val = pd.to_numeric(val, errors="coerce")
    value = val.to_numpy(dtype=np.float64)
    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(value)
    return lat[mask], lon[mask], value[mask]


def lat_lon_to_tensor(lat: np.ndarray, lon: np.ndarray) -> torch.Tensor:
    """``(N,)`` lat/lon → ``torch.float`` tensor ``(N, 2)`` with columns ``[lat, lon]``."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    return torch.from_numpy(np.column_stack([lat, lon])).float()


def embeddings_cache_path(task_name: str, backend: str) -> Path:
    """``~/data/prediction_tasks/{task_name}/embeddings_{backend}.pt``."""
    return DEFAULT_PREDICTION_TASKS_DIR / task_name / f"embeddings_{backend.lower()}.pt"


def load_embeddings(
    latlons: torch.Tensor,
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    task_name: str | None = None,
) -> torch.Tensor:
    """
    Load or compute embedding rows for ``latlons`` ``(N, 2)`` (``[lat, lon]`` per row).

    Uses the generators in ``embeddings/generate_embeddings.py``. If ``task_name`` is set,
    reads/writes ``embeddings_{backend}.pt`` under that task folder when row count matches
    (see :func:`embeddings_cache_path`).
    """
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
        ee.Initialize(project='avid-poet-486323-a3')
        emb = generate_aef_embeddings(latlons)
    else:
        raise ValueError(f"Unknown backend {backend!r}; use 'satclip', 'geoclip', or 'aef'")

    if path is not None:
        torch.save(emb, path)
    return emb


# Backward-compatible name used in older snippets / notebooks
load_or_generate_embeddings = load_embeddings


def get_task_embeddings(
    task_name: str,
    load_lat_lon_value: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    backend: str = "satclip",
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Embeddings at each ``(lat, lon)`` from ``load_lat_lon_value(**kwargs)`` (values ignored)."""
    lat, lon, _ = load_lat_lon_value(**kwargs)
    return load_embeddings(
        lat_lon_to_tensor(lat, lon),
        backend,
        device=device,
        force=force,
        task_name=task_name,
    )


def load_task_embeddings_and_values(
    task_name: str,
    load_lat_lon_value: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    backend: str = "satclip",
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Aligned ``lat``, ``lon``, task ``value``, and embedding rows for ``load_lat_lon_value(**kwargs)``."""
    lat, lon, value = load_lat_lon_value(**kwargs)
    emb = load_embeddings(
        lat_lon_to_tensor(lat, lon),
        backend,
        device=device,
        force=force,
        task_name=task_name,
    )
    return lat, lon, value, emb
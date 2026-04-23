"""All prediction tasks in one place.

File naming convention (under ~/data/prediction_tasks/):
  CSV tasks        → {task}/{task}.csv   (columns: lat_col, lon_col, value_col)
  Shapefile tasks  → {task}/{task}.shp   (joined against all_WORLD_100000_lat_lon.csv)

air_temp falls back to a Figshare download when no local CSV is found.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import (
    DEFAULT_PREDICTION_TASKS_DIR,
    lat_lon_to_tensor,
    load_csv,
    load_embeddings,
    load_shapefile,
)

# ── task configs ────────────────────────────────────────────────────────────
# CSV tasks: value_col, log_transform, lat_col, lon_col
_CSV_TASKS: dict[str, tuple[str | None, bool, str, str]] = {
    "elevation":   ("elevation", False, "lat", "lon"),
    "nightlights": ("luminosity", True,  "lat", "lon"),
    "pop_density": ("population", True,  "lat", "lon"),
    "treecover":   ("treecover",  False, "lat", "lon"),
    "air_temp":    ("meanT",      False, "Lat", "Lon"),  # value_col auto-detected; Figshare fallback
}

# Shapefile tasks: value_col
_SHP_TASKS: dict[str, str] = {
    "biome":      "BIOME_NUM",
    "ecoregion":  "ECO_ID",
    "countries":  "ISO_A3",
}

ALL_TASKS = list(_CSV_TASKS) + list(_SHP_TASKS)


# ── loading ──────────────────────────────────────────────────────────────────

def load_lat_lon_value(
    task: str,
    *,
    data_dir: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(lat, lon, value)`` for *task*."""
    root = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_PREDICTION_TASKS_DIR / task

    if task in _SHP_TASKS:
        return load_shapefile(root, value_col=_SHP_TASKS[task])

    value_col, log_transform, lat_col, lon_col = _CSV_TASKS[task]
    csv_dir = root

    lat, lon, value = load_csv(csv_dir, value_col=value_col, lat_col=lat_col, lon_col=lon_col)

    if log_transform:
        value = np.log1p(np.maximum(value, 0.0))
    return lat, lon, value


def load_dataset(
    task: str,
    backend: str = "satclip",
    *,
    device: str = "cpu",
    force: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Return ``(lat, lon, value, embeddings)`` for *task*."""
    lat, lon, value = load_lat_lon_value(task, **kwargs)
    emb = load_embeddings(
        lat_lon_to_tensor(lat, lon),
        backend,
        device=device,
        force=force,
        task_name=task,
    )
    emb_arr = np.asarray(emb)
    value_arr = np.asarray(value)
    if np.issubdtype(value_arr.dtype, np.number):
        mask = np.isfinite(value_arr) & (value_arr != -9999) & np.isfinite(emb_arr).all(axis=1)
    else:
        mask = np.ones(len(value_arr), dtype=bool) & np.isfinite(emb_arr).all(axis=1)
    return {
        'lat': lat[mask],
        'lon': lon[mask],
        'value': value[mask],
        'embeddings': emb[mask],
    }

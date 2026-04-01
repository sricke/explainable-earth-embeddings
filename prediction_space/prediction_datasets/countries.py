"""Country labels from Natural Earth admin-0 polygons + fixed lat/lon grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import (
    DEFAULT_PREDICTION_TASKS_DIR,
    get_task_embeddings,
    load_shapefile_value_at_lat_lon_csv,
    load_task_embeddings_and_values,
)

DEFAULT_COUNTRIES_SHP = (
    DEFAULT_PREDICTION_TASKS_DIR / "countries" / "ne_10m_admin_0_countries.shp"
)

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def load_lat_lon_value(
    *,
    shapefile: Path | str | None = None,
    lat_lon_csv: Path | str | None = None,
    value_col: str = "ISO_A3",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ``lat`` / ``lon`` from ``~/data/prediction_tasks/all_WORLD_100000_lat_lon.csv``
    (or ``lat_lon_csv``), then assign a country attribute from
    ``~/data/prediction_tasks/countries/ne_10m_admin_0_countries.shp`` (or ``shapefile``).

    Default label column is ``ISO_A3`` (strings). Use e.g. ``ADM0_A3`` or ``UN_A3`` if you
    prefer another Natural Earth field; numeric columns are returned as ``float64``.
    """
    shp = Path(shapefile).expanduser() if shapefile is not None else DEFAULT_COUNTRIES_SHP
    return load_shapefile_value_at_lat_lon_csv(
        shp,
        value_col=value_col,
        lat_lon_csv=lat_lon_csv,
    )


def get_embeddings(
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> torch.Tensor:
    return get_task_embeddings(
        "countries",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )


def load_embeddings_and_values(
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    return load_task_embeddings_and_values(
        "countries",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )

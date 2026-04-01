"""Ecoregions: fixed lat/lon grid + polygon labels."""

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

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def load_lat_lon_value(
    *,
    data_dir: Path | str | None = None,
    lat_lon_csv: Path | str | None = None,
    value_col: str = "ECO_ID",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ``lat`` / ``lon`` from ``~/data/prediction_tasks/all_WORLD_100000_lat_lon.csv``
    (or ``lat_lon_csv``), then assign ``value_col`` from the ecoregion shapefile under ``data_dir``.

    Default ``data_dir`` is ``~/data/prediction_tasks/ecoregion``.
    """
    shp_dir = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_PREDICTION_TASKS_DIR / "ecoregion"
    return load_shapefile_value_at_lat_lon_csv(
        shp_dir,
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
        "ecoregion",
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
        "ecoregion",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )

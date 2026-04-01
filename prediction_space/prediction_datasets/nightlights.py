"""Nighttime lights (luminosity) outcomes at point locations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .utils import (
    DEFAULT_PREDICTION_TASKS_DIR,
    get_task_embeddings,
    load_outcomes_sampled_csv,
    load_task_embeddings_and_values,
)

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def load_lat_lon_value(
    *,
    data_dir: Path | str | None = None,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ``lat``, ``lon``, ``luminosity`` from ``~/data/prediction_tasks/nightlights`` (or ``data_dir``).

    If ``log_transform`` is True, values are replaced with ``log1p(max(value, 0))`` (natural log).
    """
    path = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_PREDICTION_TASKS_DIR / "nightlights"
    lat, lon, value = load_outcomes_sampled_csv(path, value_col="luminosity")
    if log_transform:
        value = np.log1p(np.maximum(value, 0.0))
    return lat, lon, value


def get_embeddings(
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> torch.Tensor:
    return get_task_embeddings(
        "nightlights",
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
        "nightlights",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )

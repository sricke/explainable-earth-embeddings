"""Elevation outcomes sampled at point locations."""

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

# Linear probing: holdout fraction and RNG seed for row indices (see linear_probing.utils.load_embeddings_and_labels).
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def load_lat_lon_value(*, data_dir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``lat``, ``lon``, ``elevation`` from ``~/data/prediction_tasks/elevation`` (or ``data_dir``)."""
    path = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_PREDICTION_TASKS_DIR / "elevation"
    return load_outcomes_sampled_csv(path, value_col="elevation")


def get_embeddings(
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> torch.Tensor:
    return get_task_embeddings(
        "elevation",
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
        "elevation",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )

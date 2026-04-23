"""Global air temperature / precipitation point samples (SatCLIP / Figshare source)."""

from __future__ import annotations

from pathlib import Path
from urllib import request

import io
import numpy as np
import pandas as pd
import torch

from .utils import (
    DEFAULT_PREDICTION_TASKS_DIR,
    get_task_embeddings,
    load_outcomes_sampled_csv,
    load_task_embeddings_and_values,
)

_FIGSHARE_URL = "https://springernature.figshare.com/ndownloader/files/12609182"

TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42


def get_air_temp_data(pred="temp", norm_y=True, norm_x=True):
    """
    Download and process the Global Air Temperature dataset
    (https://www.nature.com/articles/sdata2018246; SatCLIP notebook).

    Parameters
    ----------
    pred
        ``\"temp\"`` or ``\"prec\"`` — which column is the prediction target.
    norm_y, norm_x
        If True, divide by column maxima.

    Returns
    -------
    coords, x, y
        ``coords`` is ``(N, 2)`` with columns ``lon``, ``lat``; ``x`` and ``y`` are 1-D float arrays.
    """
    url_open = request.urlopen(_FIGSHARE_URL)
    inc = np.array(pd.read_csv(io.StringIO(url_open.read().decode("utf-8"))))
    coords = inc[:, :2]
    if pred == "temp":
        y = inc[:, 4].reshape(-1)
        x = inc[:, 5]
    else:
        y = inc[:, 5].reshape(-1)
        x = inc[:, 4]
    if norm_y:
        y = y / y.max()
    if norm_x:
        x = x / x.max()

    return coords.astype(np.float64), x.astype(np.float64), y.astype(np.float64)


def load_lat_lon_value(
    *,
    data_dir: Path | str | None = None,
    pred: str = "temp",
    norm_y: bool = False,
    norm_x: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load air temperature (or precipitation) samples as ``lat``, ``lon``, ``value``.

    Uses ``~/data/prediction_tasks/air_temp`` if it contains a single ``*.csv`` (or
    ``outcomes_sampled_*.csv``). Otherwise downloads the Figshare CSV.

    The remote table stores coordinates as ``lon``, ``lat`` in the first two columns; this function
    returns ``lat``, ``lon``, ``value`` in that order.
    """
    root = Path(data_dir).expanduser() if data_dir is not None else DEFAULT_PREDICTION_TASKS_DIR / "air_temp"
    if root.is_file():
        return load_outcomes_sampled_csv(root, lat_col="Lat", lon_col="Lon", glob_pattern="*.csv")
    if root.is_dir():
        csvs = sorted(root.glob("*.csv"))
        if len(csvs) == 1:
            return load_outcomes_sampled_csv(csvs[0], lat_col="Lat", lon_col="Lon", glob_pattern="*.csv")
        if len(csvs) > 1:
            raise ValueError(f"Multiple CSVs in {root}; specify one file path.")

    _coords, _x, y_t = get_air_temp_data(pred=pred, norm_y=norm_y, norm_x=norm_x)
    lon = _coords[:, 0]
    lat = _coords[:, 1]
    value = y_t.reshape(-1)
    return lat, lon, value


def get_embeddings(
    backend: str = "satclip",
    *,
    device: str = "cuda",
    force: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Location embeddings for the same ``(lat, lon)`` as :func:`load_lat_lon_value` (cached per task)."""
    return get_task_embeddings(
        "air_temp",
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
    """``lat``, ``lon``, outcome, and embedding matrix with one row per location."""
    return load_task_embeddings_and_values(
        "air_temp",
        load_lat_lon_value,
        backend=backend,
        device=device,
        force=force,
        **kwargs,
    )

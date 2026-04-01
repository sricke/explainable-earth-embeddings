"""Load embeddings + probe targets for one dataset at a time.

Each task module's ``load_embeddings_and_values`` delegates to
``prediction_datasets.utils.load_task_embeddings_and_values``, which calls
:func:`prediction_datasets.utils.load_embeddings` (cached ``.pt`` files per task/backend).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prediction_datasets import (
    air_temp,
    biome,
    countries,
    ecoregion,
    elevation,
    nightlights,
    pop_density,
    treecover,
)
from prediction_datasets.utils import train_test_row_indices

EMBEDDING_BACKENDS: dict[str, str] = {
    "GeoCLIP": "geoclip",
    "SatCLIP": "satclip",
    #"AEF": "aef",
}

REGRESSION_DATASETS: dict[str, bool] = {
    "Elevation": True,
    "Population": True,
    "Nightlights": True,
    "Treecover": True,
    "Air_temp": True,
}

CLASSIFICATION_DATASETS: dict[str, bool] = {
    "Biome": True,
    "Ecoregion": True,
    "Countries": True,
}

_REGRESSION_MODULES: dict[str, Any] = {
    "Elevation": elevation,
    "Population": pop_density,
    "Nightlights": nightlights,
    "Treecover": treecover,
    "Air_temp": air_temp,
}

_CLASSIFICATION_MODULES: dict[str, Any] = {
    "Biome": biome,
    "Ecoregion": ecoregion,
    "Countries": countries,
}


DEFAULT_TEST_SIZE = 0.2
DEFAULT_SPLIT_RANDOM_STATE = 42


def _probe_split_params(mod: Any) -> tuple[float, int]:
    ts = getattr(mod, "TEST_SIZE", DEFAULT_TEST_SIZE)
    rs = getattr(mod, "SPLIT_RANDOM_STATE", DEFAULT_SPLIT_RANDOM_STATE)
    return float(ts), int(rs)


def _dataset_module(dataset_name: str) -> Any:
    if dataset_name in _REGRESSION_MODULES:
        return _REGRESSION_MODULES[dataset_name]
    if dataset_name in _CLASSIFICATION_MODULES:
        return _CLASSIFICATION_MODULES[dataset_name]
    known = sorted(_REGRESSION_MODULES | _CLASSIFICATION_MODULES)
    raise KeyError(f"Unknown dataset {dataset_name!r}. Expected one of {known}")


def _tensor_to_numpy(emb: torch.Tensor) -> np.ndarray:
    return emb.detach().cpu().numpy().astype(np.float32, copy=False)


def _values_to_probe_targets(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=np.float32)
    codes, _ = pd.factorize(s, sort=False)
    return codes.astype(np.float32)


def load_embeddings_and_labels(
    dataset_name: str,
    *,
    embedding_models: Iterable[str] | None = None,
    device: str = "cuda",
    force: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load **one** prediction dataset: embeddings and targets.

    Train/test row indices follow each task module's ``TEST_SIZE`` and
    ``PROBE_SPLIT_RANDOM_STATE`` (see e.g. ``prediction_datasets.elevation``).

    Parameters
    ----------
    dataset_name
        Key such as ``\"Elevation\"`` or ``\"Biome\"`` (see ``REGRESSION_DATASETS`` / ``CLASSIFICATION_DATASETS``).
    embedding_models
        Subset of ``EMBEDDING_BACKENDS`` keys (e.g. ``[\"SatCLIP\"]``). Default: all models.

    Returns
    -------
    latlon
        ``(N, 2)`` array of ``[lat, lon]`` per row.
    embeddings
        Maps e.g. ``\"SatCLIP\"`` -> ``(N, D)`` float32.
    targets
        Length-``N`` probe targets (float32 regression or factorized codes for classification).
    train_idx, test_idx
        Integer indices into rows ``0..N-1`` for the holdout split.
    """
    names = (
        list(EMBEDDING_BACKENDS.keys())
        if embedding_models is None
        else list(embedding_models)
    )
    unknown = set(names) - set(EMBEDDING_BACKENDS)
    if unknown:
        raise KeyError(f"Unknown embedding_models {unknown}; allowed {list(EMBEDDING_BACKENDS)}")

    mod = _dataset_module(dataset_name)
    test_size, split_rs = _probe_split_params(mod)
    out: dict[str, np.ndarray] = {}
    lat = lon = None
    value: np.ndarray | None = None
    for emb_name in names:
        backend = EMBEDDING_BACKENDS[emb_name]
        lat, lon, v, emb_t = mod.load_embeddings_and_values(
            backend=backend, device=device, force=force
        )
        out[emb_name] = _tensor_to_numpy(emb_t)
        value = v
    assert lat is not None and lon is not None and value is not None
    latlon = np.column_stack([lat, lon])
    y = _values_to_probe_targets(value)
    n = int(y.shape[0])
    train_idx, test_idx = train_test_row_indices(
        n, test_size=test_size, random_state=split_rs
    )
    return latlon, out, y, train_idx, test_idx

"""Prediction tasks: load ``(lat, lon, value, embeddings)`` from ``~/data/prediction_tasks/<name>``."""

from .datasets import ALL_TASKS, load_lat_lon_value, load_dataset
from .generate_all_embeddings import generate_all_embeddings
from .utils import (
    DEFAULT_LAT_LON_GRID_CSV,
    DEFAULT_PREDICTION_TASKS_DIR,
    load_csv,
    load_embeddings,
    load_shapefile,
    train_test_row_indices,
)

__all__ = [
    "ALL_TASKS",
    "DEFAULT_LAT_LON_GRID_CSV",
    "DEFAULT_PREDICTION_TASKS_DIR",
    "generate_all_embeddings",
    "load_csv",
    "load_dataset",
    "load_embeddings",
    "load_dataset",
    "load_lat_lon_value",
    "load_or_generate_embeddings",
    "load_outcomes_sampled_csv",
    "load_shapefile",
    "load_shapefile_value_at_lat_lon_csv",
    "train_test_row_indices",
]

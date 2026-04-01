"""Prediction tasks: load ``(lat, lon, value)`` arrays from ``~/data/prediction_tasks/<name>`` or built-in sources."""

from . import air_temp
from . import biome
from . import countries
from . import ecoregion
from . import elevation
from . import nightlights
from . import pop_density
from . import treecover
from .generate_all_embeddings import generate_all_embeddings
from .utils import (
    DEFAULT_LAT_LON_GRID_CSV,
    DEFAULT_PREDICTION_TASKS_DIR,
    get_task_embeddings,
    load_embeddings,
    load_or_generate_embeddings,
    load_outcomes_sampled_csv,
    load_shapefile_value_at_lat_lon_csv,
    load_task_embeddings_and_values,
    train_test_row_indices,
)

__all__ = [
    "DEFAULT_LAT_LON_GRID_CSV",
    "DEFAULT_PREDICTION_TASKS_DIR",
    "generate_all_embeddings",
    "get_task_embeddings",
    "load_embeddings",
    "load_or_generate_embeddings",
    "load_outcomes_sampled_csv",
    "load_shapefile_value_at_lat_lon_csv",
    "load_task_embeddings_and_values",
    "train_test_row_indices",
    "air_temp",
    "biome",
    "countries",
    "ecoregion",
    "elevation",
    "nightlights",
    "pop_density",
    "treecover",
]

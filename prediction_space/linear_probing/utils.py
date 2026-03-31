import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial import cKDTree

import sys
sys.path.append('..')
from prediction_datasets import (
    ElevationDataset, TemperatureDataset, NDVIDataset,
    PopulationDataset, NightlightsDataset, PrecipitationDataset,
    SoilCarbonDataset, LandCoverDataset, ClimateZoneDataset,
    BiomeDataset, EcoregionDataset,
)

DATA_DIR = Path.home() / "data" / "s2-100k"

REGRESSION_DATASETS = {
    "Elevation":     ElevationDataset(),
    "Temperature":   TemperatureDataset(),
    #"NDVI":          NDVIDataset(),
    "Population":    PopulationDataset(),
    "Nightlights":   NightlightsDataset(),
    "Precipitation": PrecipitationDataset(),
    "Soil Carbon":   SoilCarbonDataset(),
}

CLASSIFICATION_DATASETS = {
    #"Land Cover":   (LandCoverDataset(),  "raster"),
    "Biome":        (BiomeDataset(),       "grid"),
    "Climate Zone": (ClimateZoneDataset(), "raster"),
    "Ecoregion":    (EcoregionDataset(),   "grid"),
}


def load_embeddings(data_dir=DATA_DIR):
    """Load GeoCLIP, SatCLIP, and AEF embeddings and the lat/lon index."""
    data_dir = Path(data_dir)
    latlons = pd.read_csv(data_dir / "index.csv")[["lat", "lon"]].values
    embeddings = {
        "GeoCLIP": torch.load(data_dir / "geoclip_embeddings.pt", weights_only=True).numpy(),
        "SatCLIP": torch.load(data_dir / "satclip_embeddings.pt", weights_only=True).numpy(),
        "AEF":     torch.load(data_dir / "aef_embeddings.pt",     weights_only=True).numpy(),
    }
    return latlons, embeddings


def _lookup_raster(dataset, latlons):
    dataset._load()
    coords = [(float(lon), float(lat)) for lat, lon in latlons]
    vals = np.array([v[0] for v in dataset._src.sample(coords)], dtype=np.float32)
    nd = dataset._src.nodata
    if nd is not None:
        vals[vals == nd] = np.nan
    return vals


def _lookup_grid(dataset, latlons):
    dataset._load()
    grid_pts = np.stack([dataset._lats, dataset._lons], axis=1)
    _, idxs = cKDTree(grid_pts).query(latlons)
    return dataset._values[idxs].astype(np.float32)


def load_labels(latlons):
    """Return (regression_labels, classification_labels) dicts mapping dataset name -> np.ndarray."""
    regression_labels = {name: _lookup_raster(ds, latlons) for name, ds in REGRESSION_DATASETS.items()}
    classification_labels = {
        name: _lookup_raster(ds, latlons) if kind == "raster" else _lookup_grid(ds, latlons)
        for name, (ds, kind) in CLASSIFICATION_DATASETS.items()
    }
    return regression_labels, classification_labels

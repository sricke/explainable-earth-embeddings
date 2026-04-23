"""
Dataset for classification or regression prediction tasks.
Supported: air_temp, elevation, nightlights, pop_density, treecover, biome, ecoregion
"""

import logging
import sys
import numpy as np
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from utils import set_seed

logger = logging.getLogger(__name__)

TASK_CONFIGS = {
    "elevation":   dict(value_col="elevation", log_transform=False, lat_col="lat", lon_col="lon", task_type="regression"),
    "nightlights": dict(value_col="luminosity", log_transform=True,  lat_col="lat", lon_col="lon", task_type="regression"),
    "pop_density": dict(value_col="population", log_transform=True,  lat_col="lat", lon_col="lon", task_type="regression"),
    "treecover":   dict(value_col="treecover",  log_transform=False, lat_col="lat", lon_col="lon", task_type="regression"),
    "air_temp":    dict(value_col="meanT",       log_transform=False, lat_col="Lat", lon_col="Lon", task_type="regression"),
    "biome":       dict(value_col="BIOME_NUM",   log_transform=False, lat_col="lat", lon_col="lon", task_type="classification"),
    "ecoregion":   dict(value_col="ECO_ID",      log_transform=False, lat_col="lat", lon_col="lon", task_type="classification"),
}

REGRESSION_TASKS     = [k for k, v in TASK_CONFIGS.items() if v["task_type"] == "regression"]
CLASSIFICATION_TASKS = [k for k, v in TASK_CONFIGS.items() if v["task_type"] == "classification"]


class GeoDataset(Dataset):
    def __init__(
        self,
        task: str,
        root: Path = "~/data/prediction_tasks",
        split: str = "train",
        latlon_csv: Path | None = None,
        shp_file: Path | None = None,
        subsample: int | float | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.task = task
        assert task in TASK_CONFIGS, f"{task} not currently supported"
        self.cfg  = TASK_CONFIGS[task]
        self.root = Path(root).expanduser() / task

        split_path = self.root / f"{split}.csv"
        if not split_path.exists():
            if self.cfg["task_type"] == "classification":
                raw = self._build_classification_df(latlon_csv, shp_file)
            else:
                raw = pd.read_csv(next(iter(self.root.glob("*.csv"))))
            self._save_train_test_split(raw)

        self.df = pd.read_csv(split_path)

        val_col = self.cfg["value_col"]
        if self.cfg["log_transform"]:
            self.df[val_col] = np.log1p(np.maximum(self.df[val_col], 0.0))

        mask = self.df[val_col].notna() & np.isfinite(self.df[val_col].astype(float))
        self.df = self.df[mask].reset_index(drop=True)

        if subsample is not None:
            n = int(subsample * len(self.df)) if isinstance(subsample, float) else subsample
            self.df = self.df.sample(n=min(n, len(self.df)), random_state=seed).reset_index(drop=True)

    @property
    def latlons(self) -> np.ndarray:
        return self.df[[self.cfg["lat_col"], self.cfg["lon_col"]]].to_numpy(dtype=np.float32)

    @property
    def values(self) -> np.ndarray:
        return self.df[self.cfg["value_col"]].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        latlon = torch.tensor([row[self.cfg["lat_col"]], row[self.cfg["lon_col"]]], dtype=torch.float32)
        val    = torch.tensor(row[self.cfg["value_col"]], dtype=torch.float32)
        return latlon, val

    def _build_classification_df(self, latlon_csv: Path | None, shp_file: Path | None) -> pd.DataFrame:
        import geopandas as gpd
        from shapely.geometry import Point

        lat_col = self.cfg["lat_col"]
        lon_col = self.cfg["lon_col"]
        val_col = self.cfg["value_col"]

        if latlon_csv is None:
            latlon_csv = next(iter(self.root.glob("*.csv")))
        if shp_file is None:
            shp_file = next(iter(self.root.glob("*.shp")))

        latlon_csv = Path(latlon_csv)
        shp_file   = Path(shp_file)

        logger.info(f"Building {self.task} labels via spatial join: {latlon_csv.name} × {shp_file.name}")

        pts_df = pd.read_csv(latlon_csv)
        geometry = [Point(lon, lat) for lon, lat in zip(pts_df[lon_col], pts_df[lat_col])]
        pts_gdf  = gpd.GeoDataFrame(pts_df, geometry=geometry, crs="EPSG:4326")

        poly_gdf = gpd.read_file(shp_file).to_crs("EPSG:4326")
        joined   = gpd.sjoin(pts_gdf, poly_gdf[[val_col, "geometry"]], how="left", predicate="within")

        result = joined[[lat_col, lon_col, val_col]].copy()
        before = len(result)
        result = result.dropna(subset=[val_col])
        dropped = before - len(result)
        if dropped:
            logger.warning(f"Dropped {dropped} points with no matching polygon")

        result[val_col] = result[val_col].astype(int)
        return result.reset_index(drop=True)

    def _save_train_test_split(self, all_data: pd.DataFrame, val_size=0.1, test_size=0.1, random_state=42):
        train, test = train_test_split(all_data, test_size=test_size, random_state=random_state)
        train, val  = train_test_split(train,    test_size=val_size,  random_state=random_state)
        for name, df in [("train", train), ("val", val), ("test", test)]:
            df.to_csv(self.root / f"{name}.csv", index=False)
            logger.info(f"Saved {name} split ({len(df)} rows) → {self.root / f'{name}.csv'}")

    def plot(self, color=None, s=5):
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots(1, figsize=(6, 3))
        Basemap(projection="cyl", resolution="c", ax=ax).drawcoastlines()
        coords = self.df[[self.cfg["lon_col"], self.cfg["lat_col"]]].to_numpy()
        ax.scatter(coords[:, 0], coords[:, 1], c=color, s=s, alpha=0.7)
        return fig


def build_dataloaders(task, root="~/data/prediction_tasks", batch_size=512, num_workers=0, seed=42):
    train_dataset = GeoDataset(task, root=root, split="train")
    val_dataset   = GeoDataset(task, root=root, split="val")
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                          worker_init_fn=lambda wid: set_seed(seed + wid), generator=g)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl

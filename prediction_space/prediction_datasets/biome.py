"""Biome dataset from RESOLVE Ecoregions 2017 (WWF-based, 14 biomes)."""
from pathlib import Path

import numpy as np

from .utils import get_data_dir, download_file, extract_zip, rasterize_geodataframe

# RESOLVE 2017 Ecoregions shapefile (publicly available)
_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"

BIOME_NAMES = {
    1: "Tropical & Subtropical Moist Broadleaf Forests",
    2: "Tropical & Subtropical Dry Broadleaf Forests",
    3: "Tropical & Subtropical Coniferous Forests",
    4: "Temperate Broadleaf & Mixed Forests",
    5: "Temperate Conifer Forests",
    6: "Boreal Forests/Taiga",
    7: "Tropical & Subtropical Grasslands, Savannas & Shrublands",
    8: "Temperate Grasslands, Savannas & Shrublands",
    9: "Flooded Grasslands & Savannas",
    10: "Montane Grasslands & Shrublands",
    11: "Tundra",
    12: "Mediterranean Forests, Woodlands & Scrub",
    13: "Deserts & Xeric Shrublands",
    14: "Mangroves",
}


class BiomeDataset:
    """
    Global biome classification (14 classes) from RESOLVE Ecoregions 2017.

    Rasterized from the WWF-based shapefile to a 0.1° global grid on first load.
    __getitem__ returns ((lat, lon), biome_id).
    """

    DEFAULT_FILENAME = "Ecoregions2017.zip"
    SHAPEFILE_NAME = "Ecoregions2017.shp"

    TASK_NAME = "biome"

    def __init__(self, file_path=None, data_dir=None, resolution=0.1, download=False):
        self._data_dir = get_data_dir(data_dir) / self.TASK_NAME
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._shp_path = Path(file_path) if file_path else self._data_dir / self.SHAPEFILE_NAME
        self._resolution = resolution
        self._lats = self._lons = self._values = None
        if download and not self._shp_path.exists():
            self.download()

    def verify_download(self):
        """Check the shapefile exists and can be opened. Returns True if valid."""
        if not self._shp_path.exists():
            print(f"[MISSING]  {self._shp_path.name}")
            return False
        try:
            import geopandas as gpd
            gdf = gpd.read_file(self._shp_path, rows=1)
            assert "BIOME_NUM" in gdf.columns
            print(f"[OK]       {self._shp_path.name}")
            return True
        except Exception as e:
            print(f"[CORRUPT]  {self._shp_path.name} — {e}")
            return False

    def _load(self):
        if self._lats is not None:
            return
        if not self._shp_path.exists():
            raise FileNotFoundError(
                f"{self._shp_path} not found. Call .download() first."
            )
        import geopandas as gpd
        print("Loading and rasterizing biome shapefile (first time, may take a minute)...")
        gdf = gpd.read_file(self._shp_path)
        self._lats, self._lons, self._values = rasterize_geodataframe(
            gdf, value_col="BIOME_NUM", resolution=self._resolution
        )

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)
        shp = data_dir / self.SHAPEFILE_NAME

        if shp.exists():
            print(f"{self.SHAPEFILE_NAME} already exists, skipping.")
        else:
            zip_path = download_file(_URL, data_dir)
            extract_zip(zip_path, data_dir)

        if not shp.exists():
            raise FileNotFoundError(f"Expected {shp} after extraction.")
        self._shp_path = shp
        return shp

    def __len__(self):
        self._load()
        return len(self._values)

    def __getitem__(self, idx):
        self._load()
        return (float(self._lats[idx]), float(self._lons[idx])), int(self._values[idx])

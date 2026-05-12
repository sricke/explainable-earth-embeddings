import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from paths import DATA_ROOT, SHAPEFILE

SEED = 0
N = 100_000
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)
OUT = DATA_ROOT / "dense_grid" / "dense_grid.csv"

rng = np.random.default_rng(SEED)

land_union = gpd.read_file(SHAPEFILE).to_crs("EPSG:4326").union_all()

lat_min_sin = np.sin(np.radians(LAT_RANGE[0]))
lat_max_sin = np.sin(np.radians(LAT_RANGE[1]))

collected = []
batch = N * 10  # land ≈ 30% of surface, so one pass is almost always enough

while sum(len(p) for p in collected) < N:
    lons = rng.uniform(LON_RANGE[0], LON_RANGE[1], batch)
    lats = np.degrees(np.arcsin(rng.uniform(lat_min_sin, lat_max_sin, batch)))
    gdf = gpd.GeoDataFrame(
        {"lat": lats, "lon": lons},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    collected.append(gdf[gdf.geometry.within(land_union)][["lat", "lon"]])

df = pd.concat(collected).iloc[:N].reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Saved {len(df)} points -> {OUT}")

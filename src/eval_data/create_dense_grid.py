"""
Code adapted from the create_grid_dense code https://codeocean.com/capsule/6456296/tree/v2
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

SEED = 42
N = 100_000
LAT_RANGE = (-90, 90)
LON_RANGE = (-180, 180)

parser = argparse.ArgumentParser()
parser.add_argument("--shapefile", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

rng = np.random.default_rng(SEED)

land_union = gpd.read_file(args.shapefile).to_crs("EPSG:4326").union_all()

# Sample uniformly in sin(lat) space
lat_min_sin = np.sin(np.radians(LAT_RANGE[0]))
lat_max_sin = np.sin(np.radians(LAT_RANGE[1]))

collected = []
batch = N * 10  # land is approx 30% of surface, so one pass is almost always enough

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

args.out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(args.out, index=False)
print(f"Saved {len(df)} points -> {args.out}")

import math
import pandas as pd
import glob, os
import numpy as np

for path in glob.glob(os.path.expanduser("~/data/git-10M/*.parquet")):
    df = pd.read_parquet(path)

    # remove bad rows
    df = df.dropna(subset=["Google_location"])
    df = df[df["Google_location"].str.count("_") == 2]

    # split z_x_y
    zxy = df["Google_location"].str.split("_", expand=True).astype(int)
    z, x, y = zxy[0].values, zxy[1].values, zxy[2].values

    n = 2 ** z

    lon = x / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2.0 * y / n))))

    df["lon"] = lon
    df["lat"] = lat

    df.to_parquet(path)

    print(f"Processed {path}")
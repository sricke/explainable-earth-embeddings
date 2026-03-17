import pandas as pd 
import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
from utils import geo_to_mercator, mercator_to_geo
from loguru import logger

fp = "data/sentinel2_good_files.csv"
good_sentinel_files = pd.read_csv(fp)
logger.debug(f"good_sentinel_files.shape: {good_sentinel_files.shape}")

def get_colrow(col, row):
    return f"{col}-{row}"

def f(col, row):
    lon, lat = mercator_to_geo((col, row), pixels=1, zoom=13)
    lon2, lat2 = mercator_to_geo((col+1, row+1), pixels=1, zoom=13)
    mid_lon, mid_lat = (lon+lon2)/2, (lat+lat2)/2
    return (mid_lon,mid_lat)
# sinr['col-row'] = sinr.apply(lambda x: f(x['longitude'], x['latitude']), axis=1)
# sinr["col"] = sinr.apply(lambda x: x["col-row"][0], axis=1)
# sinr["row"] = sinr.apply(lambda x: x["col-row"][1], axis=1)
# good_sentinel_files["col-row"] = good_sentinel_files.apply(lambda x: get_colrow(x["col"], x["row"]), axis=1)
logger.debug(f"Assigning lon-lat to each entry")
good_sentinel_files['lon-lat'] = good_sentinel_files.apply(lambda x: f(x['col'], x['row']), axis=1)
logger.debug(f"Separating lon-lat to different columns")
good_sentinel_files["longitude"] = good_sentinel_files.apply(lambda x: x["lon-lat"][0], axis=1)
good_sentinel_files["latitude"] = good_sentinel_files.apply(lambda x: x["lon-lat"][1], axis=1)

logger.debug(f"Saving file...")
good_sentinel_files.to_csv("data/sentinel2_good_files_withlatlon.csv", index=False)

logger.debug(f"Filtering to year 2022...")
same_year_data = good_sentinel_files[good_sentinel_files["year"]==2022]

logger.debug(f"Removing duplicate files and shuffling")
df = same_year_data.iloc[np.random.permutation(len(same_year_data))]    # shuffle dataframe
deduped = df.groupby("fp").head(1)    # remove duplicate locations

logger.debug(f"Choosing 100k")
sampled_100k = deduped[:100000]
logger.debug(f"sampled_100k: {sampled_100k.shape}")

logger.debug(f"Saving file")
sampled_100k.to_csv("data/deduped100k_sentinel2_good_files_withlatlonmid.csv", index=False)
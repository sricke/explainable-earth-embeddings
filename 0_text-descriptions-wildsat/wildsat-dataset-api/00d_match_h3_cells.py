# Reference for h3 cells resolution: https://h3geo.org/docs/core-library/restable/
# Based on math of longest diagonal of hexagon:
# res 8 longest diagonal is approx 1.06km (100 sentinel pixels) -- means patch size cannot be less than 128 
# res 9 longest diagonal is approx 0.403km (40 sentinel pixels) -- means patch size cannot be less than 64
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import random
import datetime
from tqdm import tqdm
from utils import geo_to_mercator, mercator_to_geo

import cv2
# from h3.unstable import vect
# import h3.api.numpy_int as h3
import h3
from loguru import logger


def assign_h3_cell(lat, lon, resolution=6):
    h3_cell = h3.geo_to_h3(lat, lon, resolution)    # uses h3 version 3.7.7
    return h3_cell

logger.debug("Loading sentinel and geo data")
deduped_geo_data = pd.read_csv("deduped_filtered_geo_prior_data.csv")
sentinel_df = pd.read_csv("deduped_sentinel2_good_files_withlatlon.csv")
logger.debug(f"deduped_geo_data: {deduped_geo_data.shape}")
logger.debug(f"sentinel_df: {sentinel_df.shape}")

# Add res8 and res9
logger.debug(f"Adding resolutions 8 and 9 to sentinel")
sentinel_df["h3_res8"] = sentinel_df.apply(lambda x: assign_h3_cell(x["latitude"], x["longitude"], resolution=8), axis=1)
sentinel_df["h3_res9"] = sentinel_df.apply(lambda x: assign_h3_cell(x["latitude"], x["longitude"], resolution=9), axis=1)
logger.debug(f"Adding resolutions 8 and 9 to geo data")
deduped_geo_data["h3_res8"] = deduped_geo_data.apply(lambda x: assign_h3_cell(x["latitude"], x["longitude"], resolution=8), axis=1)
deduped_geo_data["h3_res9"] = deduped_geo_data.apply(lambda x: assign_h3_cell(x["latitude"], x["longitude"], resolution=9), axis=1)

logger.debug(f"Merging based on res8")
merged_df_res8 = sentinel_df.merge(deduped_geo_data, how = 'inner', on = ['h3_res8'])
merged_df_res8["taxon_id-h3_res8"] =  merged_df_res8[["taxon_id", "h3_res8"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
logger.debug(f"merged_df_res8: {merged_df_res8.shape} saving to file merged_df_res8.csv")
merged_df_res8.to_csv("merged_df_res8.csv", index=False)


logger.debug(f"Merging based on res9")
merged_df_res9 = sentinel_df.merge(deduped_geo_data, how = 'inner', on = ['h3_res9'])
merged_df_res9["taxon_id-h3_res9"] =  merged_df_res9[["taxon_id", "h3_res9"]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
logger.debug(f"merged_df_res9: {merged_df_res9.shape} saving to file merged_df_res9.csv")
merged_df_res9.to_csv("merged_df_res9.csv", index=False)
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
from loguru import logger

random.seed(123)
np.random.seed(123)

sentinel2_dir = "/datasets/ai/allenai/satlas_pretrain/sentinel2"    # data from SatlasPretrain dataset
"""
fp = "/datasets/ai/allenai/satlas_pretrain/metadata/good_images_lowres_all.json"
with open(fp, "r") as f:    # contains few black/empty pixels and clouds
    good_images_lowres_all = json.load(f)


all_files = []
ctr_dir_not_exist = 0
for col, row, dir_name in tqdm(good_images_lowres_all):
    if not os.path.exists(os.path.join(sentinel2_dir, dir_name)):
        ctr_dir_not_exist += 1
        continue
    if len(dir_name.split("_")) < 3:
        logger.debug(f"{dir_name} not processed")
        continue
    datetime_str = dir_name.split("_")[2]
    date_obj = datetime.datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")
    if not os.path.exists(os.path.join(sentinel2_dir, dir_name, "tci")):
        print(f"{dir_name} doesn't have tci folder")
        ctr_dir_not_exist += 1
        continue
    filename = f"{col}_{row}.png"
    fp = os.path.join(dir_name, "tci", filename)
    all_files.append([fp, datetime_str, date_obj.year, date_obj.month, date_obj.day, col, row])
logger.debug(f"Didn't find {ctr_dir_not_exist} good imgs")

# for dir_name in tqdm(os.listdir(sentinel2_dir)):
#     if len(dir_name.split("_")) < 3:
#         logger.debug(f"{dir_name} not processed")
#         continue
#     datetime_str = dir_name.split("_")[2]
#     date_obj = datetime.datetime.strptime(datetime_str, "%Y%m%dT%H%M%S")
#     if not os.path.exists(os.path.join(sentinel2_dir, dir_name, "tci")):
#         print(f"{dir_name} doesn't have tci folder")
#         continue
#     for filename in  os.listdir(os.path.join(sentinel2_dir, dir_name, "tci")):
#         # print(filename)
#         fp = os.path.join(dir_name, "tci", filename)
#         col, row = filename.split(".")[0].split("_")
#         all_files.append([fp, datetime_str, date_obj.year, date_obj.month, date_obj.day, col, row])
#         # break
#     # break

df = pd.DataFrame(all_files, columns=["fp","datetime", "year", "month", "day", "col", "row"])
df.to_csv("sentinel2_good_files.csv", index=False)
"""

df = pd.read_csv("sentinel2_good_files.csv")
logger.debug(f"df: {df.shape}")
logger.debug("Reading sinr data")
sinr = pd.read_csv("sinr_with_mercator.csv")
sinr["col"] = (sinr["col"]).astype(int)
sinr["row"] = (sinr["row"]).astype(int)
df["col"] = (df["col"]).astype(int)
df["row"] = (df["row"]).astype(int)


logger.debug("Matching sinr data with sentinel")
merged_df = df.merge(sinr, how = 'inner', on = ['col', 'row'])
logger.debug(f"merged_df: {merged_df.shape}")

logger.debug("Filtering to same year")
same_year_data = merged_df[merged_df["year_x"]==merged_df["year_y"]]
logger.debug(f"same_year_data: {same_year_data.shape}")

logger.debug("Deduplicating same location observations")
same_year_data.to_csv("common_sinr_goodsentinel2_locyear.csv", index=False)
df = same_year_data.iloc[np.random.permutation(len(same_year_data))]    # shuffle dataframe
samp = df.groupby("col-row").head(1)    # remove duplicate locations
logger.debug(f"deduped same year data samp: {samp.shape}")

samp.to_csv("matched_sinr_goodsentinel2_deduped.csv", index=False)
logger.debug("Saved results to: matched_sinr_goodsentinel2_deduped.csv")
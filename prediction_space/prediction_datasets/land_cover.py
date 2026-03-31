"""ESA WorldCover 2021 global Map download via AWS S3."""

from pathlib import Path
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import rasterio
from rasterio.merge import merge

from .utils import RasterDataset, get_data_dir

LAND_COVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


class LandCoverDataset(RasterDataset):
    """
    ESA WorldCover 10m 2021 v200 global land cover (Map layer).

    Downloads all 3x3 degree tiles from the public AWS bucket and
    (optionally) merges into a global GeoTIFF.
    """

    DEFAULT_FILENAME = "ESA_WorldCover_2021_global.tif"
    AWS_BUCKET = "esa-worldcover"
    AWS_PREFIX = "v200/2021/map/"

    TASK_NAME = "land_cover"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def _read(self):
        from .utils import read_raster
        lats, lons, values = read_raster(self._file_path, nodata=0)
        return lats, lons, values.astype(np.int32)

    def _list_s3_tiles(self):
        """
        List all Map tile keys in the bucket.
        Uses boto3 with unsigned config so no credentials needed.
        """
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        paginator = s3.get_paginator("list_objects_v2")

        keys = []
        for page in paginator.paginate(Bucket=self.AWS_BUCKET, Prefix=self.AWS_PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("_Map.tif"):
                    keys.append(key)
        return keys

    def _download_tile(self, key, out_dir):
        """
        Download a single S3 key via HTTPS.
        """
        url = f"https://{self.AWS_BUCKET}.s3.eu-central-1.amazonaws.com/{key}"
        filename = Path(key).name
        out_path = out_dir / filename

        if out_path.exists():
            return out_path

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        return out_path

    def _download_all_tiles(self, data_dir, max_workers=8):
        tile_dir = data_dir / "tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        keys = self._list_s3_tiles()

        paths = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for p in tqdm(ex.map(lambda k: self._download_tile(k, tile_dir), keys), total=len(keys)):
                paths.append(p)

        return paths

    def _merge_tiles(self, tile_paths, output_path):
        print("Merging tiles into global raster...")

        srcs = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = merge(srcs)

        out_meta = srcs[0].meta.copy()
        out_meta.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "compress": "lzw",
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        for src in srcs:
            src.close()

    def download(self, data_dir=None, merge_tiles=True):
        data_dir = get_data_dir(data_dir or self._data_dir)
        out_tif = data_dir / self.DEFAULT_FILENAME

        if merge_tiles and out_tif.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            self._file_path = out_tif
            return out_tif

        tile_paths = self._download_all_tiles(data_dir)

        if not merge_tiles:
            self._file_path = tile_paths[0]
            return tile_paths

        self._merge_tiles(tile_paths, out_tif)

        self._file_path = out_tif
        return out_tif
"""Mean annual temperature from WorldClim v2.1 (ERA5-Land/GHCN, 10 arc-minute)."""
from pathlib import Path

import numpy as np
import rasterio

from .utils import RasterDataset, download_file, extract_zip, get_data_dir

_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip"


class TemperatureDataset(RasterDataset):
    """
    Mean annual temperature in °C from WorldClim v2.1.

    Monthly averages are averaged to produce the annual mean on first download.
    __getitem__ returns ((lat, lon), temperature_celsius).
    """

    DEFAULT_FILENAME = "worldclim_mean_annual_temperature.tif"
    TASK_NAME = "temperature"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)
        out_path = data_dir / self.DEFAULT_FILENAME
        if out_path.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            self._file_path = out_path
            return out_path

        zip_path = download_file(_URL, data_dir)
        extract_zip(zip_path, data_dir)

        monthly = sorted(data_dir.glob("wc2.1_10m_tavg_??.tif"))
        if not monthly:
            raise FileNotFoundError("Monthly temperature files not found after extraction.")

        arrays = []
        profile = None
        for f in monthly:
            with rasterio.open(f) as src:
                if profile is None:
                    profile = src.profile.copy()
                arr = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                arrays.append(arr)

        mean_temp = np.nanmean(np.stack(arrays), axis=0)
        nodata_val = -9999.0
        mean_temp[np.isnan(mean_temp)] = nodata_val
        profile.update(dtype=rasterio.float32, nodata=nodata_val, count=1)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mean_temp, 1)

        self._file_path = out_path
        return out_path

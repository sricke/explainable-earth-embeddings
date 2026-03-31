"""Elevation dataset from WorldClim v2.1 (SRTM-based, 10 arc-minute)."""
from pathlib import Path

from .utils import RasterDataset, download_file, extract_zip, get_data_dir

_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_elev.zip"


class ElevationDataset(RasterDataset):
    """
    Global elevation in metres from WorldClim v2.1 (SRTM-based).

    __getitem__ returns ((lat, lon), elevation_m).
    """

    DEFAULT_FILENAME = "wc2.1_10m_elev.tif"
    TASK_NAME = "elevation"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)
        tif = data_dir / self.DEFAULT_FILENAME

        if tif.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
        else:
            zip_path = download_file(_URL, data_dir)
            extract_zip(zip_path, data_dir)
            
        if not tif.exists():
            raise FileNotFoundError(f"Expected {tif} after extraction.")
        self._file_path = tif
        return tif

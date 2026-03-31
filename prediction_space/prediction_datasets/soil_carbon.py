"""Soil organic carbon from SoilGrids 250m (ISRIC) via WCS at 0.1° resolution."""
from pathlib import Path

from .utils import RasterDataset, download_file, get_data_dir


class SoilCarbonDataset(RasterDataset):
    """
    Soil organic carbon (dg/kg) in the 0–5 cm layer from SoilGrids (ISRIC).

    Downloaded at ~0.1° resolution via the SoilGrids WCS API.
    __getitem__ returns ((lat, lon), soc_dg_per_kg).
    """

    DEFAULT_FILENAME = "soilgrids_soc_0-5cm_mean.tif"
    TASK_NAME = "soil_carbon"
    
    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)
        tif = data_dir / self.DEFAULT_FILENAME

        if tif.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            self._file_path = tif
            return tif

        import requests

        url = "https://maps.isric.org/mapserv"
        params = {
            "map": "/map/soc.map",
            "SERVICE": "WCS",
            "VERSION": "2.0.1",
            "REQUEST": "GetCoverage",
            "COVERAGEID": "soc_0-5cm_mean",
            "FORMAT": "image/tiff",
            "SUBSET": ["Lat(-90,90)", "Long(-180,180)"],
        }

        r = requests.get(url, params=params, stream=True)
        r.raise_for_status()

        with open(tif, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

        self._file_path = tif
        return tif

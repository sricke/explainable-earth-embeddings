"""Köppen-Geiger climate zone from Beck et al. 2018 (1 km, 30 classes)."""
from pathlib import Path

import numpy as np

from .utils import RasterDataset, download_file, extract_zip, get_data_dir

# Beck et al. (2018) present-day classification, 1 km resolution
# Published in Scientific Data: https://doi.org/10.1038/sdata.2018.214
# Hosted on figshare (present.zip contains present_reclassified.tif)
_URL = "https://ndownloader.figshare.com/files/12407516"

KOPPEN_CLASSES = {
    1: "Af", 2: "Am", 3: "Aw", 4: "BWh", 5: "BWk",
    6: "BSh", 7: "BSk", 8: "Csa", 9: "Csb", 10: "Csc",
    11: "Cwa", 12: "Cwb", 13: "Cwc", 14: "Cfa", 15: "Cfb",
    16: "Cfc", 17: "Dsa", 18: "Dsb", 19: "Dsc", 20: "Dsd",
    21: "Dwa", 22: "Dwb", 23: "Dwc", 24: "Dwd", 25: "Dfa",
    26: "Dfb", 27: "Dfc", 28: "Dfd", 29: "ET", 30: "EF",
}


class ClimateZoneDataset(RasterDataset):
    """
    Global Köppen-Geiger climate zone classification (30 classes) from Beck et al. 2018.

    __getitem__ returns ((lat, lon), class_id) where class_id maps to KOPPEN_CLASSES.
    """

    DEFAULT_FILENAME = "present_reclassified.tif"
    TASK_NAME = "climate_zone"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def _read(self):
        from .utils import read_raster
        lats, lons, values = read_raster(self._file_path, nodata=0)
        return lats, lons, values.astype(np.int32)

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)
        out_tif = data_dir / self.DEFAULT_FILENAME
        if out_tif.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            self._file_path = out_tif
            return out_tif

        zip_path = download_file(_URL, data_dir, filename="koppen_present.zip")
        extract_zip(zip_path, data_dir)

        # Beck et al. zip contains 'present.tif' or 'present_reclassified.tif'
        candidates = list(data_dir.glob("*present*.tif"))
        if not candidates:
            raise FileNotFoundError(
                f"Expected a present*.tif in {data_dir} after extraction. "
                f"Files found: {list(data_dir.iterdir())}"
            )
        tif = candidates[0]
        if tif != out_tif:
            tif.rename(out_tif)
        self._file_path = out_tif
        return out_tif

"""Nighttime lights from VIIRS DNB Annual Composite (EOG, Colorado School of Mines)."""
from pathlib import Path

from .utils import RasterDataset, get_data_dir

# EOG VIIRS annual composites require free registration at:
# https://eogdata.mines.edu/products/vnl/
# After downloading, point file_path to the average_masked .tif file.
_DOWNLOAD_PAGE = "https://eogdata.mines.edu/products/vnl/"


class NightlightsDataset(RasterDataset):
    """
    Global nighttime lights (nW/cm²/sr) from VIIRS DNB Annual Composite.

    Download the annual average_masked GeoTIFF from EOG (free registration):
      https://eogdata.mines.edu/products/vnl/
    Then instantiate with file_path pointing to the downloaded .tif.

    __getitem__ returns ((lat, lon), radiance).
    """

    DEFAULT_FILENAME = "VNL_2024_annual_average_masked.tif"
    TASK_NAME = "nightlights"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def download(self, data_dir=None):
        self._file_path = data_dir / self.DEFAULT_FILENAME
        if self._file_path.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            return self._file_path 
        else:
            raise RuntimeError(
                "VIIRS nighttime lights requires free registration.\n"
                f"Download the annual average_masked GeoTIFF from:\n  {_DOWNLOAD_PAGE}\n"
                "Then pass file_path= to NightlightsDataset()."
            )

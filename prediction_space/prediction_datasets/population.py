"""Population density from WorldPop 2020 (1 km, aggregated)."""
from .utils import RasterDataset, download_file, get_data_dir

_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020/"
    "2020/0_Mosaicked/ppp_2020_1km_Aggregated.tif"
)


class PopulationDataset(RasterDataset):
    """
    Global population density (persons/km²) from WorldPop 2020.

    __getitem__ returns ((lat, lon), population_per_km2).
    """

    DEFAULT_FILENAME = "ppp_2020_1km_Aggregated.tif"
    TASK_NAME = "population"

    def __init__(self, file_path=None, data_dir=None, download=False):
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def download(self, data_dir=None):
        data_dir = get_data_dir(data_dir or self._data_dir)

        tif = data_dir / self.DEFAULT_FILENAME

        if tif.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
        else:
            tif = download_file(_URL, data_dir)

        self._file_path = tif
        return tif

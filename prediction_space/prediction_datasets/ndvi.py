"""NDVI dataset from MODIS MOD13C2 (0.05° monthly composite, LAADSWEB)."""

from pathlib import Path
import os
import requests
from bs4 import BeautifulSoup
import subprocess

from .utils import RasterDataset, download_file, get_data_dir

_BASE_URL = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD13C2.061/"


class NDVIDataset(RasterDataset):
    """
    Global NDVI from MODIS MOD13C2 (0.05° monthly composite).

    Requires NASA Earthdata credentials for download.
    __getitem__ returns ((lat, lon), ndvi) where NDVI is in [-1, 1].
    """

    DEFAULT_FILENAME = "MOD13C2_NDVI.tif"
    TASK_NAME = "ndvi"

    def __init__(self, file_path=None, data_dir=None, download=False):  
        super().__init__(self.TASK_NAME, file_path=file_path, data_dir=data_dir, download=download)

    def _read(self):
        from .utils import read_raster
        # NDVI stored as int16 scaled by 0.0001, nodata=-3000
        return read_raster(self._file_path, nodata=-3000, scale=0.0001)

    def _list_hdf_for_year(self, year, auth):
        """
        List HDF files for a given year directory on the LAADS archive.
        """
        year_url = f"{_BASE_URL}{year}/"
        r = requests.get(year_url, auth=auth)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # find all HDF file links
        files = [a.text for a in soup.find_all("a") if a.text.endswith(".hdf")]
        if not files:
            raise FileNotFoundError(f"No .hdf found in {year_url}")
        return [year_url + fn for fn in files]

    def download(
        self,
        year="2023",
        username=None,
        password=None,
        data_dir=None,
    ):
        """
        Download MOD13C2 NDVI from NASA LAADSWEB for a given year.

        Parameters
        ----------
        year : str
            Year subdirectory under the MOD13C2.061 archive (e.g., "2023").
        username : str, optional
            NASA Earthdata username.
        password : str, optional
            NASA Earthdata password.
        data_dir : str or Path, optional
            Directory to save data.
        """
        data_dir = get_data_dir(data_dir or self._data_dir)
        username = username or os.environ.get("EARTHDATA_USERNAME")
        password = password or os.environ.get("EARTHDATA_PASSWORD")

        if not username or not password:
            raise RuntimeError(
                "NASA Earthdata credentials required.\n"
                "Register at https://urs.earthdata.nasa.gov/ then pass "
                "username= and password= or set EARTHDATA_USERNAME / "
                "EARTHDATA_PASSWORD environment variables."
            )

        out_path = data_dir / self.DEFAULT_FILENAME
        if out_path.exists():
            print(f"{self.DEFAULT_FILENAME} already exists, skipping.")
            self._file_path = out_path
            return out_path

        auth = (username, password)

        # list all HDFs for the year
        hdf_urls = self._list_hdf_for_year(year, auth)

        # pick first monthly HDF (you can loop all later if needed)
        hdf_url = hdf_urls[0]
        print(f"Downloading {hdf_url} ...")
        hdf_path = download_file(hdf_url, data_dir, auth=auth)

        # Convert HDF NDVI subdataset to GeoTIFF
        subdataset = f'HDF4_EOS:EOS_GRID:"{hdf_path}":MOD_Grid_monthly_CMG_VI:CMG 0.05 Deg Monthly NDVI'
        subprocess.run(
            ["gdal_translate", "-of", "GTiff", subdataset, str(out_path)],
            check=True,
        )

        self._file_path = out_path
        return out_path
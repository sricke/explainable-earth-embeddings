import zipfile
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm

DEFAULT_DATA_DIR = Path.home() / "data" / "prediction_tasks"


def get_data_dir(data_dir=None):
    path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url, dest_dir, filename=None, auth=None):
    """Download a file to dest_dir, skipping if already present."""
    dest_dir = get_data_dir(dest_dir)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0] or "download"
    dest_path = dest_dir / filename
    if dest_path.exists():
        print(f"{filename} already exists, skipping.")
        return dest_path
    print(f"Downloading {filename} ...")
    resp = requests.get(url, stream=True, auth=auth, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest_path


def extract_zip(zip_path, dest_dir=None):
    """Extract a zip file, returning the destination directory."""
    zip_path = Path(zip_path)
    dest = Path(dest_dir) if dest_dir else zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return dest


def read_raster(file_path, nodata=None, scale=1.0):
    """Read a raster, returning (lats, lons, values) 1-D arrays of valid pixels."""
    with rasterio.open(file_path) as src:
        data = src.read(1).astype(np.float32)
        nd = nodata if nodata is not None else src.nodata
        mask = np.isfinite(data)
        if nd is not None:
            mask &= data != nd
        rows, cols = np.where(mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lats = np.asarray(ys, dtype=np.float32)
        lons = np.asarray(xs, dtype=np.float32)
        values = data[rows, cols] * scale
    return lats, lons, values


def rasterize_geodataframe(gdf, value_col, resolution=0.1):
    """
    Rasterize a GeoDataFrame to a regular lat/lon grid.
    Returns (lats, lons, values) 1-D arrays for cells with value > 0.
    """
    from rasterio.features import rasterize as rio_rasterize

    west, south, east, north = -180.0, -90.0, 180.0, 90.0
    width = int((east - west) / resolution)
    height = int((north - south) / resolution)
    transform = from_bounds(west, south, east, north, width, height)

    shapes = (
        (geom, int(val))
        for geom, val in zip(gdf.geometry, gdf[value_col])
        if geom is not None and not np.isnan(float(val))
    )
    grid = rio_rasterize(shapes, out_shape=(height, width), transform=transform,
                         fill=0, dtype=np.int32)

    rows, cols = np.where(grid > 0)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lats = np.asarray(ys, dtype=np.float32)
    lons = np.asarray(xs, dtype=np.float32)
    values = grid[rows, cols]
    return lats, lons, values


class RasterDataset:
    """Base class for raster-based datasets. __getitem__ returns (lat, lon) -> value."""

    DOWNLOAD_URL = None
    DEFAULT_FILENAME = None

    def __init__(self, task_name, file_path=None, data_dir=None, download=False):
        self._task_name = task_name
        self._data_dir = Path(get_data_dir(data_dir)) / task_name
        self._file_path = Path(file_path) if file_path else self._data_dir / self.DEFAULT_FILENAME
        self._src = None  # rasterio dataset handle
        if download and not self._file_path.exists():
            self.download()

    def _load(self):
        if self._src is None:
            if not self._file_path.exists():
                raise FileNotFoundError(
                    f"{self._file_path} not found. Call .download() first."
                )
            self._src = rasterio.open(self._file_path)

    def download(self, data_dir=None):
        raise NotImplementedError

    def verify_download(self):
        """Check the file exists and can be opened. Returns True if valid."""
        if not self._file_path.exists():
            print(f"[MISSING]  {self._file_path.name}")
            return False
        try:
            with rasterio.open(self._file_path) as src:
                _ = src.meta
            print(f"[OK]       {self._file_path.name}")
            return True
        except Exception as e:
            print(f"[CORRUPT]  {self._file_path.name} — {e}")
            return False

    def __getitem__(self, latlon: tuple):
        """Return raster value at (lat, lon) using rasterio.sample."""
        self._load()
        lat, lon = latlon

        # rasterio expects (x, y) = (lon, lat)
        val = next(self._src.sample([(lon, lat)]))[0]

        # Handle nodata
        if self._src.nodata is not None and val == self._src.nodata:
            return np.nan

        return float(val)

    def close(self):
        """Optional: manually close dataset."""
        if self._src is not None:
            self._src.close()
            self._src = None

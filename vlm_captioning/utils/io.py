"""CSV I/O for caption runs (lat/lon, paths, text)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd


CAPTION_COLUMNS = ("lat", "lon", "image_path", "caption")


def save_captions(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    index: bool = False,
) -> None:
    """Persist a dataframe with lat, lon, image path, and caption to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(CAPTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    df[list(CAPTION_COLUMNS)].to_csv(path, index=index)


def load_captions(path: Union[str, Path]) -> pd.DataFrame:
    """Load captions CSV (expects lat, lon, image_path, caption)."""
    df = pd.read_csv(path)
    for c in CAPTION_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Expected column {c!r} in {path}")
    return df


def load_captions_with_lat_lon(
    path: Union[str, Path],
    *,
    crs_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Load captions and ensure lat/lon are numeric (optional CRS hint reserved for GeoJSON joins)."""
    df = load_captions(path)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if crs_hint:
        # Hook for future pyproj / geodataframe alignment
        pass
    return df

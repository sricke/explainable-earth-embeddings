"""Satellite tile discovery and download via STAC (Sentinel-2, optional NAIP for higher resolution)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import requests
from rasterio.plot import reshape_as_image

# Public STAC catalog (no API key). For NAIP (US, ~1 m) use Planetary Computer STAC if Earth Search has no NAIP.
DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"
FALLBACK_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

CollectionName = Literal[
    "sentinel-2-l2a",
    "naip",
]


@dataclass
class TileSpec:
    """One tile to download: WGS84 bbox and time window."""

    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    datetime_range: str  # e.g. "2024-06-01T00:00:00Z/2024-09-01T00:00:00Z"
    collection: CollectionName = "sentinel-2-l2a"


def _stac_search(
    catalog_url: str,
    collection: str,
    bbox: Tuple[float, float, float, float],
    datetime_range: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    url = catalog_url.rstrip("/") + "/search"
    body = {
        "collections": [collection],
        "bbox": list(bbox),
        "datetime": datetime_range,
        "limit": limit,
    }
    r = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("features", [])


def _pick_visual_href(item: Dict[str, Any]) -> Optional[str]:
    assets = item.get("assets") or {}
    for key in ("visual", "preview", "thumbnail", "rendered_preview"):
        a = assets.get(key)
        if a and a.get("href"):
            return a["href"]
    # Sentinel-2 sometimes exposes red/green/blue
    for key in ("B04", "B03", "B02"):
        if key in assets and assets[key].get("href"):
            return None  # multi-band path below
    return None


def _stack_rgb_from_assets(item: Dict[str, Any]) -> Optional[np.ndarray]:
    """Stack B04,B03,B02 (or common naming) to HWC uint8 if individual COGs exist."""
    assets = item.get("assets") or {}
    keys = [
        ("B04", "B03", "B02"),
        ("red", "green", "blue"),
    ]
    hrefs = None
    for r, g, b in keys:
        if r in assets and g in assets and b in assets:
            hrefs = (assets[r]["href"], assets[g]["href"], assets[b]["href"])
            break
    if not hrefs:
        return None
    arrs = []
    for href in hrefs:
        with rasterio.open(href) as src:
            a = src.read(1).astype(np.float32)
            # scale reflectance to 0-255 if needed
            if a.max() <= 1.5:
                a = np.clip(a * 255.0, 0, 255)
            else:
                a = np.clip(a, 0, 255)
            arrs.append(a)
    # nearest common shape
    shapes = [x.shape for x in arrs]
    if len(set(shapes)) != 1:
        target = min(shapes, key=lambda s: s[0] * s[1])
        resized = []
        for a in arrs:
            if a.shape != target:
                # simple subsample to min shape
                y0 = (a.shape[0] - target[0]) // 2
                x0 = (a.shape[1] - target[1]) // 2
                a = a[y0 : y0 + target[0], x0 : x0 + target[1]]
            resized.append(a)
        arrs = resized
    rgb = np.stack(arrs, axis=-1).astype(np.uint8)
    return rgb


def _read_href_to_rgb(href: str) -> np.ndarray:
    with rasterio.open(href) as src:
        arr = src.read()
    if arr.shape[0] >= 3:
        img = reshape_as_image(arr[:3])
    else:
        img = reshape_as_image(arr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, np.percentile(img, 99))
        img = (img / (img.max() + 1e-6) * 255).astype(np.uint8)
    return img


def download_tile(
    spec: TileSpec,
    output_path: Union[str, Path],
    *,
    stac_url: str = DEFAULT_STAC_URL,
    collection: Optional[CollectionName] = None,
    prefer_high_res: bool = True,
) -> Path:
    """
    Download a satellite RGB tile for ``spec.bbox`` and time range.

    **Resolution:** Sentinel-2 L2A is ~10 m in the RGB/visual product. For **higher**
    native resolution (sub-meter over the US), set ``prefer_high_res=True`` and use
    collection ``naip`` when your AOI is in North America; otherwise Sentinel-2 is used.

    Args:
        spec: Bbox + datetime window.
        output_path: Path to write ``.png`` or ``.tif`` (PNG recommended for VLMs).
        stac_url: STAC API base URL.
        collection: Override collection (defaults from spec or NAIP when prefer_high_res).
        prefer_high_res: If True, try NAIP first (US), then fall back to Sentinel-2.

    Returns:
        Path to written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    col = collection or spec.collection
    if prefer_high_res and collection is None:
        # Try NAIP (high-res aerial, US) on Earth Search, then Planetary Computer, then Sentinel-2.
        feats = _stac_search(stac_url, "naip", spec.bbox, spec.datetime_range, limit=3)
        if not feats and stac_url == DEFAULT_STAC_URL:
            feats = _stac_search(FALLBACK_STAC_URL, "naip", spec.bbox, spec.datetime_range, limit=3)
        if not feats:
            feats = _stac_search(stac_url, "sentinel-2-l2a", spec.bbox, spec.datetime_range, limit=3)
            col = "sentinel-2-l2a"
        else:
            col = "naip"
    else:
        feats = _stac_search(stac_url, col, spec.bbox, spec.datetime_range, limit=5)

    if not feats:
        raise RuntimeError(
            f"No STAC items for bbox={spec.bbox} datetime={spec.datetime_range} collection={col}"
        )

    item = feats[0]
    href = _pick_visual_href(item)
    if href:
        rgb = _read_href_to_rgb(href)
    else:
        rgb = _stack_rgb_from_assets(item)
        if rgb is None:
            raise RuntimeError("Could not resolve visual or RGB assets from STAC item")

    ext = output_path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        from PIL import Image

        Image.fromarray(rgb).save(output_path)
    elif ext == ".tif":
        # single-band uint8 stack
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=rgb.shape[0],
            width=rgb.shape[1],
            count=3,
            dtype=rgb.dtype,
        ) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")

    meta_path = output_path.with_suffix(output_path.suffix + ".stac_item.json")
    meta_path.write_text(json.dumps(item, indent=2))
    return output_path


class SatelliteTileIndex:
    """Simple manifest of downloaded tiles (paths + bbox + source collection)."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SatelliteTileIndex":
        data = json.loads(Path(path).read_text())
        return cls(data.get("tiles", []))

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps({"tiles": self.records}, indent=2))

    def append(
        self,
        image_path: str,
        bbox: Sequence[float],
        collection: str,
        datetime_range: str,
    ) -> None:
        self.records.append(
            {
                "image_path": image_path,
                "bbox": list(bbox),
                "collection": collection,
                "datetime": datetime_range,
            }
        )

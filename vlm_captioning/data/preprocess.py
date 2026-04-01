"""Batch preprocessing: crop, resize, optional super-resolution before captioning."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple, Union

import pandas as pd
from PIL import Image

from utils.image import super_resolve_image


def _center_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    w, h = img.size
    tw, th = size
    left = max(0, (w - tw) // 2)
    top = max(0, (h - th) // 2)
    return img.crop((left, top, left + tw, top + th))


def preprocess_one(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    crop_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    super_resolve: bool = False,
    sr_scale: int = 2,
    sr_mode: Literal["bicubic", "lanczos", "kornia"] = "lanczos",
) -> Path:
    """
    Load, optionally center-crop, optionally upscale (for finer appearance), optionally resize, save.

    For **higher effective resolution**, enable ``super_resolve`` (after crop) or prefer
    NAIP / commercial imagery at download time; see ``data.dataset.download_tile``.
    """
    image_path, output_path = Path(image_path), Path(output_path)
    img = Image.open(image_path).convert("RGB")
    if crop_size is not None:
        if center_crop:
            img = _center_crop(img, crop_size)
        else:
            img = img.crop((0, 0, min(crop_size[0], img.width), min(crop_size[1], img.height)))
    if super_resolve:
        img = super_resolve_image(img, scale=sr_scale, mode=sr_mode)
    if resize is not None:
        img = img.resize(resize, Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def preprocess_batch(
    image_paths: Iterable[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    crop_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = True,
    resize: Optional[Tuple[int, int]] = None,
    super_resolve: bool = False,
    sr_scale: int = 2,
    pattern: str = "{stem}_prep{suffix}",
) -> pd.DataFrame:
    """
    Preprocess many images; returns a dataframe with original and output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for p in image_paths:
        p = Path(p)
        out = output_dir / pattern.format(stem=p.stem, suffix=p.suffix)
        preprocess_one(
            p,
            out,
            crop_size=crop_size,
            center_crop=center_crop,
            resize=resize,
            super_resolve=super_resolve,
            sr_scale=sr_scale,
        )
        rows.append({"source_path": str(p), "image_path": str(out)})
    return pd.DataFrame(rows)


def build_manifest(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    path_col: str = "image_path",
) -> pd.DataFrame:
    """Keep only lat, lon, image_path columns for the captioning loader."""
    return df[[lat_col, lon_col, path_col]].copy()

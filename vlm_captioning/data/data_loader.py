"""Load satellite images and tabular metadata (lat, lon, image path) for captioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils.image import load_and_super_resolve


class SatelliteImageDataset(Dataset):
    """
    Rows with ``lat``, ``lon``, and ``image_path`` (preprocessed or raw).

    ``transform`` receives a PIL image after optional ``load_and_super_resolve``.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        transform: Optional[Callable[[Image.Image], Any]] = None,
        super_resolve: bool = False,
        sr_scale: int = 2,
    ) -> None:
        required = {"lat", "lon", "image_path"}
        if not required.issubset(frame.columns):
            raise ValueError(f"DataFrame must contain columns {required}")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.super_resolve = super_resolve
        self.sr_scale = sr_scale

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.frame.iloc[idx]
        path = Path(row["image_path"])
        img, _meta = load_and_super_resolve(
            path,
            super_resolve=self.super_resolve,
            scale=self.sr_scale,
        )
        if self.transform is not None:
            img = self.transform(img)
        return {
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "image_path": str(path),
            "image": img,
        }


def load_manifest_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV with at least lat, lon, image_path."""
    df = pd.read_csv(path)
    for c in ("lat", "lon", "image_path"):
        if c not in df.columns:
            raise ValueError(f"Missing column {c!r} in {path}")
    return df


def make_dataloader(
    frame: pd.DataFrame,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = False,
    transform: Optional[Callable[[Image.Image], Any]] = None,
    super_resolve: bool = False,
    sr_scale: int = 2,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> DataLoader:
    """Build a ``DataLoader`` over :class:`SatelliteImageDataset`."""
    ds = SatelliteImageDataset(
        frame,
        transform=transform,
        super_resolve=super_resolve,
        sr_scale=sr_scale,
    )

    def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # PIL images stay as list when batch_size > 1 (VLMs often encode one-by-one)
        return {
            "lat": torch.tensor([b["lat"] for b in batch], dtype=torch.float64),
            "lon": torch.tensor([b["lon"] for b in batch], dtype=torch.float64),
            "image_path": [b["image_path"] for b in batch],
            "image": [b["image"] for b in batch],
        }

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or default_collate,
    )

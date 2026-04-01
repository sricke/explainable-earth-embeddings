"""Single-image, batch, and full-dataset captioning."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_loader import SatelliteImageDataset, make_dataloader
from models.prompt_engineering import get_prompts
from models.vlm_model import VLMModel
from utils.io import save_captions


class CaptionGenerator:
    """Runs a :class:`~models.vlm_model.VLMModel` over files or a :class:`SatelliteImageDataset`."""

    def __init__(
        self,
        model: VLMModel,
        *,
        prompt: Optional[str] = None,
        prompt_style: str = "default",
        merge_prompts: bool = True,
    ) -> None:
        self.model = model
        if prompt is not None:
            self._prompt = prompt
        else:
            p = get_prompts(style=prompt_style, merge=merge_prompts)
            self._prompt = p if isinstance(p, str) else "\n\n".join(p)

    def caption_image(
        self,
        image: Union[Image.Image, str, Path],
        *,
        max_new_tokens: int = 512,
    ) -> str:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        return self.model.generate_caption(image, self._prompt, max_new_tokens=max_new_tokens)

    def caption_batch_paths(
        self,
        paths: List[Union[str, Path]],
        lats: List[float],
        lons: List[float],
        *,
        max_new_tokens: int = 512,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        if not (len(paths) == len(lats) == len(lons)):
            raise ValueError("paths, lats, and lons must have the same length")
        rows = []
        it = zip(paths, lats, lons)
        if show_progress:
            it = tqdm(list(it), desc="Captioning")
        for p, lat, lon in it:
            cap = self.caption_image(Path(p), max_new_tokens=max_new_tokens)
            rows.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "image_path": str(p),
                    "caption": cap,
                }
            )
        return pd.DataFrame(rows)

    def caption_dataframe(
        self,
        df: pd.DataFrame,
        *,
        max_new_tokens: int = 512,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self.caption_batch_paths(
            list(df["image_path"]),
            list(df["lat"]),
            list(df["lon"]),
            max_new_tokens=max_new_tokens,
            show_progress=show_progress,
        )

    def caption_dataloader(
        self,
        loader: DataLoader,
        *,
        max_new_tokens: int = 512,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Consume batches from :func:`data.data_loader.make_dataloader` (batch_size often 1)."""
        rows: List[dict] = []
        it = loader
        if show_progress:
            it = tqdm(loader, desc="Captioning")
        for batch in it:
            images = batch["image"]
            if isinstance(images, torch.Tensor):
                raise TypeError("Expected PIL images in batch['image']; use default collate or list of PIL.")
            lats = batch["lat"]
            lons = batch["lon"]
            paths = batch["image_path"]
            if isinstance(lats, torch.Tensor):
                lats = lats.tolist()
            if isinstance(lons, torch.Tensor):
                lons = lons.tolist()
            if not isinstance(images, list):
                images = [images]
            n = len(paths)
            for i in range(n):
                img = images[i]
                lat = float(lats[i])
                lon = float(lons[i])
                p = paths[i]
                if not isinstance(img, Image.Image):
                    raise TypeError("Each image must be PIL.Image.Image")
                cap = self.model.generate_caption(img, self._prompt, max_new_tokens=max_new_tokens)
                rows.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "image_path": str(p),
                        "caption": cap,
                    }
                )
        return pd.DataFrame(rows)

    def caption_dataset(
        self,
        dataset: SatelliteImageDataset,
        *,
        max_new_tokens: int = 512,
        batch_size: int = 1,
        num_workers: int = 0,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        loader = make_dataloader(
            dataset.frame,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            super_resolve=dataset.super_resolve,
            sr_scale=dataset.sr_scale,
        )
        return self.caption_dataloader(loader, max_new_tokens=max_new_tokens, show_progress=show_progress)

    def run_and_save(
        self,
        df: pd.DataFrame,
        output_csv: Union[str, Path],
        **kwargs,
    ) -> pd.DataFrame:
        out = self.caption_dataframe(df, **kwargs)
        save_captions(out, output_csv)
        return out


def caption_single_file(
    model: VLMModel,
    image_path: Union[str, Path],
    *,
    prompt_style: str = "default",
    max_new_tokens: int = 512,
) -> str:
    """Convenience: one image path -> caption string."""
    gen = CaptionGenerator(model, prompt_style=prompt_style)
    return gen.caption_image(image_path, max_new_tokens=max_new_tokens)

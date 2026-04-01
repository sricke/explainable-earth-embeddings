"""Image loading and optional upscaling for lower-native-resolution sensors (e.g. Sentinel-2 10 m)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_pil_image(path: Union[str, Path]) -> Image.Image:
    """Load an image as RGB PIL.Image."""
    path = Path(path)
    img = Image.open(path).convert("RGB")
    return img


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def _tensor01_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255.0).round().astype(np.uint8))


def super_resolve_image(
    image: Image.Image,
    scale: int = 2,
    mode: Literal["bicubic", "lanczos", "kornia"] = "bicubic",
) -> Image.Image:
    """
    Upsample an image to emulate higher effective resolution for captioning.

    Sentinel-2 RGB composites are often 10 m; bicubic/Lanczos upsampling can help VLMs
    reason over finer structure. For stronger super-resolution, plug in a dedicated
    model (e.g. Real-ESRGAN) in place of this helper.

    Args:
        image: Input RGB image.
        scale: Integer upscale factor (e.g. 2 doubles width/height).
        mode: ``bicubic`` (torch), ``lanczos`` (PIL), or ``kornia`` (differentiable bicubic).
    """
    if scale < 1:
        raise ValueError("scale must be >= 1")
    if scale == 1:
        return image.copy()

    w, h = image.size
    new_w, new_h = w * scale, h * scale

    if mode == "lanczos":
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    t = _pil_to_tensor01(image)
    if mode == "kornia":
        if kornia is None:
            raise ImportError("kornia is required for mode='kornia'")
        try:
            resize_fn = kornia.geometry.transform.resize
        except AttributeError:
            resize_fn = None
        if resize_fn is not None:
            t_up = resize_fn(t, (new_h, new_w), interpolation="bicubic", antialias=True)
        else:
            t_up = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
        return _tensor01_to_pil(t_up)

    # bicubic via torch (no extra deps)
    t_up = F.interpolate(t, size=(new_h, new_w), mode="bicubic", align_corners=False)
    return _tensor01_to_pil(t_up)


def load_and_super_resolve(
    path: Union[str, Path],
    *,
    super_resolve: bool = False,
    scale: int = 2,
    sr_mode: Literal["bicubic", "lanczos", "kornia"] = "lanczos",
) -> Tuple[Image.Image, dict]:
    """
    Load from disk; optionally upsample for higher-res appearance before VLM encoding.

    Returns the image and a small metadata dict describing what was applied.
    """
    img = load_pil_image(path)
    meta: dict = {"path": str(path), "super_resolve": False, "scale": 1}
    if super_resolve:
        img = super_resolve_image(img, scale=scale, mode=sr_mode)
        meta["super_resolve"] = True
        meta["scale"] = scale
    return img, meta

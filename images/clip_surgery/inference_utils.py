"""Shared helpers for CLIP surgery inference (GeoCLIP + SatCLIP)."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "CLIP_Surgery") not in sys.path:
    sys.path.insert(0, str(_ROOT / "CLIP_Surgery"))

import clip  # noqa: E402
import rasterio  # noqa: E402

BICUBIC = InterpolationMode.BICUBIC

_geoclip_preprocess = Compose(
    [
        Resize((224, 224), interpolation=BICUBIC),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)
RANDOM_LOCATION=[-1, -19.827, -123.225] # (id, lon, lat)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def s2_stack_to_rgb_u8(image_chw: np.ndarray) -> np.ndarray:
    """True-color (H,W,3) uint8 from Sentinel-2 stack (C,H,W) reflectance ~[0,1]."""
    x = np.moveaxis(image_chw, 0, 2)
    rgb = x[:, :, [3, 2, 1]]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def load_rgb_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_s2_sample(
    index_path: str,
    id_column: str,
    lon_column: str = "lon",
    lat_column: str = "lat",
) -> List[List[Any]]:
    df_index = pd.read_csv(index_path)
    return [
        [id_val, lon, lat]
        for id_val, lon, lat in zip(
            df_index[id_column], df_index[lon_column], df_index[lat_column]
        )
    ]


def load_image_geoclip(path: str, device: str = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return _geoclip_preprocess(image).unsqueeze(0).to(device)

def preprocessing_satclip(image: Union[np.ndarray, torch.Tensor], device: str = "cpu") -> torch.Tensor:
    # Sentinel-2 tiles come either as raw DN values (integer, ~[0, 10000]) or as
    # already-scaled reflectance (float, ~[0, 1]). Only rescale in the first case;
    # otherwise the encoder would see ~zero input and produce identical features
    # across tiles.
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    raw_dtype = image.dtype
    if np.issubdtype(raw_dtype, np.integer) or float(np.nanmax(image)) > 10.0:
        image = image / 10000.0
    image = image.astype(np.float32)
    # S2-100K ships 12 bands (B10 dropped), so pad a zero plane at index 10.
    if image.shape[0] == 12:
        b10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], b10, image[10:]], axis=0)
    rgb_img = s2_stack_to_rgb_u8(image)
    x = torch.from_numpy(image).float().to(device).unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return x, rgb_img

def load_image_sentinel(path: str, device: str = "cpu") -> tuple[torch.Tensor, np.ndarray]:
    with rasterio.open(path) as f:
        image = f.read()
    x, rgb_img = preprocessing_satclip(image, device)
    return x, rgb_img


def encode_locations(
    locations: List,
    encode_fn,
    device: str,
    *,
    satclip: bool,
) -> torch.Tensor:
    if satclip:
        locations_formatted = [[lon, lat] for _, lon, lat in locations]
    else:
        locations_formatted = [[lat, lon] for _, lon, lat in locations]
    locations_tensor = torch.tensor(
        locations_formatted,
        device=device,
        dtype=torch.float32,
    )
    location_features = encode_fn(locations_tensor).float()
    location_features = location_features / location_features.norm(dim=-1, keepdim=True)
    if location_features.dim() == 3 and location_features.shape[1] == 1:
        location_features = location_features.squeeze(1)
    return location_features


def safe_results_subdir(image_id: str) -> str:
    """Stable folder name under a results directory (basename, no separators)."""
    base = os.path.basename(str(image_id).strip())
    return base.replace(os.sep, "_").replace("\x00", "")


def surgery_similarity_patch_rows(sim: torch.Tensor) -> torch.Tensor:
    """Strip CLS when token count is ``1 + H*W``; keep rows if already ``H*W``."""
    n = sim.shape[1]
    s = int(round(float(n) ** 0.5))
    if s * s == n:
        return sim
    if n > 1 and (n - 1) == s * s:
        return sim[:, 1:, :]
    return sim


HeatmapNormalize = Literal["none", "minmax"]


def heatmap_to_uint8_for_colormap(
    hm: np.ndarray,
    normalize: HeatmapNormalize = "none",
) -> np.ndarray:
    """Convert a 2D similarity heatmap to uint8 for ``applyColorMap``.

    * ``none``: same as before inference — ``(hm * 255).astype(uint8)`` (assumes ~[0, 1]).
    * ``minmax``: per-panel linear stretch to [0, 1] then scale to [0, 255] so each panel uses
      the full colormap range (easier to compare structure across layers).
    """
    hm = np.asarray(hm, dtype=np.float64).squeeze()
    if hm.ndim != 2:
        raise ValueError(f"heatmap must be 2D for applyColorMap, got shape {hm.shape}")
    if normalize == "none":
        return (hm * 255).astype(np.uint8)
    if normalize == "minmax":
        lo, hi = float(hm.min()), float(hm.max())
        if hi - lo < 1e-12:
            return np.zeros(hm.shape, dtype=np.uint8)
        stretched = (hm - lo) / (hi - lo)
        return (stretched * 255).astype(np.uint8)
    raise ValueError(f"normalize must be 'none' or 'minmax', got {normalize!r}")


def plot_layer_similarity_grid(
    heatmaps: List[np.ndarray],
    cv2_bgr_background: np.ndarray,
    out_path: str,
    titles: Optional[List[str]] = None,
    suptitle: str = "",
    ncols: int = 4,
    dpi: int = 150,
    figsize: Optional[tuple[float, float]] = None,
    heatmap_normalize: HeatmapNormalize = "none",
    pad_inches: float = 0.1,
) -> None:
    n = len(heatmaps)
    nrows = math.ceil(n / ncols) if n else 1
    if figsize is None:
        figsize = (ncols * 3.0, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i < n:
            hm = heatmaps[i]
            vis = heatmap_to_uint8_for_colormap(hm, normalize=heatmap_normalize)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_bgr_background * 0.6 + vis * 0.4
            vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
            ax.imshow(vis)
            ax.set_title(titles[i] if titles else f"L{i}")
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi)
    plt.close(fig)


def save_surgery_maps_to_png(
    place_name: str,
    results_dir: str,
    similarity_maps_dict: Dict[str, torch.Tensor],
    cv2_img_bgr_dict: Dict[str, np.ndarray],
) -> None:
    """Write similarity heatmaps under ``results_dir / <image_id> / similarity*.png``."""
    for pid in tqdm(similarity_maps_dict.keys(), desc=f"Saving maps {place_name}"):
        sub = safe_results_subdir(pid)
        out_dir = os.path.join(results_dir, sub)
        os.makedirs(out_dir, exist_ok=True)
        similarity_map = similarity_maps_dict[pid]
        cv2_img_bgr = cv2_img_bgr_dict[pid]
        for b in range(similarity_map.shape[0]):
            vis = (similarity_map[b, :, :].numpy() * 255).astype(np.uint8)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_img_bgr * 0.6 + vis * 0.4
            vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
            plt.imshow(vis)
            plt.title("CLIP Surgery")
            plt.axis("off")
            fname = "clip_surgery.png" if similarity_map.shape[0] == 1 else f"clip_surgery_{b}.png"
            plt.savefig(
                os.path.join(out_dir, fname),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

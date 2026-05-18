from __future__ import annotations

import math
from collections import namedtuple
from typing import Any, Dict, List, Literal, Optional, Union
import random
from dataclasses import dataclass
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
import rasterio  
from pathlib import Path
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

@dataclass
class Location:
    id: str
    lon: float
    lat: float

RANDOM_LOCATION=Location(id=None, lon=-19.827, lat=-123.225)

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

def seed_everything(seed: int) -> None:
    """Fix all relevant random seeds for reproducible inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_locations(
    csv_path: str,
    lon_column: str,
    lat_column: str,
    id_column: str | None = None,
) -> List[Location]:
    df = pd.read_csv(csv_path)
    return [Location(id=df[id_column][i], lon=df[lon_column][i], lat=df[lat_column][i]) for i in range(len(df))]

def load_image_geoclip(path: str, device: str = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return _geoclip_preprocess(image).unsqueeze(0).to(device)

def load_image_sentinel(path: str, device: str = "cpu") -> tuple[torch.Tensor, np.ndarray]:
    with rasterio.open(path) as f:
        image = f.read()
    x, rgb_img = _preprocessing_satclip(image, device)
    return x, rgb_img

def load_rgb_from_images_corr(place: str, id_val: str) -> np.ndarray | None:
    """Load RGB preview PNG from images_corr/ (or Images_corr/) if present."""
    stem = Path(id_val).stem
    candidates = [
        Path(place) / "images_corr" / f"{stem}.png",
        Path(place) / "images" / f"{stem}.png",
    ]
    for p in candidates:
        if p.is_file():
           rgb = np.array(Image.open(str(p)).convert("RGB"))
           return rgb
    return None

def _preprocessing_satclip(image: Union[np.ndarray, torch.Tensor], device: str = "cpu") -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    raw_dtype = image.dtype
    # some images already come normalized from download so adding check before dividing by 10000
    if np.issubdtype(raw_dtype, np.integer) or float(np.nanmax(image)) > 10.0:
        image = image / 10000.0
    image = image.astype(np.float32)
    # S2-100K ships 12 bands (B10 dropped), so pad a zero plane at index 10.
    if image.shape[0] == 12:
        b10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], b10, image[10:]], axis=0)
    x = np.moveaxis(image, 0, 2)
    rgb = x[:, :, [3, 2, 1]] # BGR to RGB
    rgb_img = (np.clip(rgb, 0, 1)*255).astype(np.uint8) # clip to [0, 1] and then to [0, 255]
    x = torch.from_numpy(image).float().to(device)
    x = F.interpolate(x.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return x, rgb_img

def encode_locations(
    locations: List,
    encode_fn,
    device: str,
    *,
    satclip: bool,
) -> torch.Tensor:
    # flip lon and lat if necessary
    if satclip:
        locations_formatted = [[loc.lon, loc.lat] for loc in locations]
    else:
        locations_formatted = [[loc.lat, loc.lon] for loc in locations]
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


def surgery_similarity_patch_rows(sim: torch.Tensor) -> torch.Tensor:
    """Strip CLS when token count is ``1 + H*W``; keep rows if already ``H*W``."""
    n = sim.shape[1]
    s = int(round(float(n) ** 0.5))
    if s * s == n:
        return sim
    if n > 1 and (n - 1) == s * s:
        return sim[:, 1:, :]
    return sim

def _minmax_normalize(hm: np.ndarray) -> np.ndarray:
    lo, hi = float(hm.min()), float(hm.max())
    if hi - lo < 1e-12:
        return np.zeros(hm.shape, dtype=np.uint8)
    stretched = (hm - lo) / (hi - lo)
    return stretched


def plot_similarity_maps(
    heatmaps: List[np.ndarray],
    cv2_bgr_background: np.ndarray,
    out_path: str,
    titles: Optional[List[str]] = None,
    suptitle: str = "",
    ncols: int = 4,
    dpi: int = 150,
    figsize: Optional[tuple[float, float]] = None,
    heatmap_normalize: Literal["none", "minmax"] = "none",
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
            hm = np.asarray(hm, dtype=np.float64).squeeze()
            if heatmap_normalize == "minmax":
                hm = _minmax_normalize(hm)
            hm = (hm * 255).astype(np.uint8)
            vis = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
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
     
# FOR BBOXES
       
def _centered_square_box(
    cx: float,
    cy: float,
    side: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Fixed-size square centered on (cx, cy), clipped to image bounds."""
    side = max(1, int(side))
    side = min(side, img_w, img_h)
    half = side // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, side, side

def _split_mask_by_erosion(binary_mask, max_area):
    final_output = np.zeros_like(binary_mask)
    kernel = np.ones((3,3), np.uint8)
    
    # get all blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    
    # Put blobs that are too big into a "todo" list; keep the rest
    todo_stack = []
    for i in range(1, num):
        blob_mask = (labels == i).astype(np.uint8) * 255
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            todo_stack.append(blob_mask)
        else:
            final_output = cv2.bitwise_or(final_output, blob_mask)

    # Process the large blobs until they are split or small enough
    while todo_stack:
        current_blob = todo_stack.pop()
        
        # Erode once 
        eroded = cv2.erode(current_blob, kernel, iterations=1)
        
        n_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(eroded)
    
        if n_sub <= 1:
            continue

        for j in range(1, n_sub):
            child_mask = (sub_labels == j).astype(np.uint8) * 255
            child_area = sub_stats[j, cv2.CC_STAT_AREA]
            
            if child_area > max_area:
                # Still too big, put back on stack to erode again
                todo_stack.append(child_mask)
            else:
                final_output = cv2.bitwise_or(final_output, child_mask)
                
    return final_output

def get_saliency_mask(
    hm: np.ndarray,
    *,
    percentile: float,
    min_peak_fraction: float,
    open_kernel_size: int,
) -> np.ndarray:
    """
    Map similarity heatmap to binary mask based on percentile and min peak fraction.
    """
    hm = np.asarray(hm, dtype=np.float64)
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max - hm_min < 1e-12:
        return np.zeros(hm.shape, dtype=np.uint8)

    p = float(np.clip(percentile, 0.0, 1.0))
    q_thr = float(np.quantile(hm, p))
    peak_thr = hm_max * float(np.clip(min_peak_fraction, 0.0, 1.0))
    thr = max(q_thr, peak_thr)
    mask = (hm >= thr).astype(np.uint8)

    k = int(open_kernel_size)
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def boxes_from_saliency(
    original_mask: np.ndarray,
    *,
    max_area: int,
    img_h: int,
    img_w: int,
    square_side: int,
) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray]:
    """ Main function to get bounding boxes from saliency mask.
    """
    mask = original_mask.copy()
    mask = _split_mask_by_erosion(mask, max_area)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        if square_side is not None:
            bx, by, bw, bh = _centered_square_box(cx, cy, square_side, img_w, img_h)
        else:
            bx, by, bw, bh = x, y, w, h
        boxes.append((bx, by, bw, bh, int(area)))
    return boxes, mask


def draw_boxes_on_rgb(
    rgb_img: np.ndarray,
    boxes: list[tuple[int, int, int, int, int]],
    *,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on RGB image and return RGB result."""
    out_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for x, y, w, h, _ in boxes:
        cv2.rectangle(out_bgr, (x, y), (x + w, y + h), color_bgr, int(thickness))
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)



import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "CLIP_Surgery") not in sys.path:
    sys.path.insert(0, str(_root / "CLIP_Surgery"))

import torch
import cv2
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import typing as tp
import rasterio
import torch.nn.functional as F
import math

from huggingface_hub import hf_hub_download
import clip
from images.clip_surgery import get_satclip
from images.clip_surgery.geoclip_surgery.load import get_geoclip
from images.clip_surgery.inference_utils import (
    heatmap_to_uint8_for_colormap,
    sentinel_rgb_preview_stretch,
)

BICUBIC = InterpolationMode.BICUBIC

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def _s2_stack_to_rgb_u8(image_chw: np.ndarray) -> np.ndarray:
    x = np.moveaxis(image_chw, 0, 2)
    rgb = x[:, :, [3, 2, 1]]
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def load_rgb_image(image_path):
    # OpenCV expects a NumPy array, not a PIL Image object.
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

def plot_layer_similarity_grid(
    heatmaps: tp.List[np.ndarray],
    cv2_bgr_background: np.ndarray,
    out_path: str,
    titles: tp.Optional[tp.List[str]] = None,
    suptitle: str = "",
    ncols: int = 4,
    dpi: int = 150,
    heatmap_normalize: str = "none",
):
    n = len(heatmaps)
    nrows = math.ceil(n / ncols) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 3.0))
    axes_flat = np.atleast_1d(axes).ravel()
    for i in range(nrows * ncols):
        ax = axes_flat[i]
        if i < n:
            hm = heatmaps[i]
            vis = heatmap_to_uint8_for_colormap(hm, normalize=heatmap_normalize)
            vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            vis = cv2_bgr_background * 0.4 + vis * 0.6
            vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2RGB)
            ax.imshow(vis)
            ax.set_title(titles[i] if titles else f"L{i}")
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def load_s2_sample(index_path, id_column, lon_column="lon", lat_column="lat"):
    df_index = pd.read_csv(index_path)
    # Vectorized operation instead of iterrows (much faster)
    locations = [[id_val, lon, lat] for id_val, lon, lat in zip(df_index[id_column], df_index[lon_column], df_index[lat_column])]
    return locations

_geoclip_preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

def load_image_geoclip(path, device="cpu"):
    image = Image.open(path).convert("RGB")
    image = _geoclip_preprocess(image).unsqueeze(0).to(device)
    return image


def load_image_sentinel(path, device="cpu"):
    with rasterio.open(path) as f:
        image = f.read().astype(np.float32)
    image = image / 10000.0
    B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
    image = np.concatenate([image[:10], B10, image[10:]], axis=0)
    rgb_img = _s2_stack_to_rgb_u8(image)
    x = torch.from_numpy(image).float().to(device).unsqueeze(0)
    x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
    c = x.shape[1]
    mean = _CLIP_MEAN.to(device).repeat((c + 2) // 3)[:c].view(1, c, 1, 1)
    std = _CLIP_STD.to(device).repeat((c + 2) // 3)[:c].view(1, c, 1, 1)
    x = (x - mean) / std
    return x, rgb_img


def encode_locations(locations: tp.List, encode_location: tp.Callable, device):
    # Batch all locations at once instead of one by one
    locations_tensor = torch.tensor([[lon, lat] for _, lon, lat in locations], device=device)
    # Encode all locations in batch
    location_features = encode_location(locations_tensor).float()  # (N, #tokens, dim)
    # Normalize
    location_features = location_features / location_features.norm(dim=-1, keepdim=True)
    # Squeeze token dimension if needed
    if location_features.dim() == 3 and location_features.shape[1] == 1:
        location_features = location_features.squeeze(1)  # (N, dim)
    return location_features

SUPPORTED_MODELS = {
    "satclip": {
        "vit16": "microsoft/SatCLIP-ViT16-L40",
        "resnet50": "microsoft/SatCLIP-ResNet50-L40",
    },
    "geoclip": {
        "vit16": "example", # TODO: add geoclip model path
        "resnet50": "example", # TODO: add geoclip model path
    },
}


if __name__ == "__main__":
    ### Init CLIP and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--model", choices=["satclip", "geoclip"], default="satclip")
    args.add_argument("--arquitecture", choices=["vit16", "resnet50"], default="vit16")
    args.add_argument("--source_folder", type=str, default="/home/seri6958/Places")
    args.add_argument("--nosurgery", action="store_true", default=False)
    args.add_argument(
        "--layer_grid_normalize",
        choices=["none", "minmax"],
        default="none",
        help="Per-panel heatmap scaling for *_layers.png (none=legacy, minmax=full colormap per panel).",
    )
    args = args.parse_args()
    
    use_surgery = not args.nosurgery

    if args.arquitecture == "resnet50":
        raise ValueError("Resnet50 not implemented yet")
    
    if args.model == "satclip":
        try:
            satclip_model_path = SUPPORTED_MODELS["satclip"][args.arquitecture]
        except KeyError:
            raise ValueError(f"Invalid satclip arquitecture: {args.arquitecture}")
        
        model = get_satclip(
            hf_hub_download(satclip_model_path, "satclip-vit16-l40.ckpt"),
            device=device,
            surgery=use_surgery,
            return_all=True,
        )  # False for original CLIP
        model.eval()
    elif args.model == "geoclip":
        geo_model = get_geoclip(device=device, surgery=use_surgery, return_all=True)
        model, _ = clip.load("CS-ViT-L/14", device=device)
        geo_model.eval()
        model.eval()


    places_folder = args.source_folder
    data_folder = "data" if args.model == "satclip" else "images"
    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]
    
    all_texts = [
        "car", "tree", "mountain", "rock", "road", "valley", "tress", "person", "bench",
        "forest", "river", "lake", "desert", "coast", "cliff", "beach", "meadow", "canyon", "plain", "field"
    ]
    
    classes = {
        "bolivia.jpg": [
            "car", "mountain", "rock", "road", "valley"
        ],
        "demo.jpg": [
            "person", "bench"
        ]
    }
    
    with torch.no_grad():
        text_features = clip.encode_text_with_prompt_ensemble(model, all_texts, device)
    if args.model == "geoclip":
            encode_location = geo_model.location_encoder.forward
            _vision_model = geo_model.image_encoder.CLIP.vision_model
            _visual_proj = geo_model.image_encoder.CLIP.visual_projection
            _image_mlp = geo_model.image_encoder.mlp

            def encode_image(image):
                if hasattr(_vision_model, "forward_intermediates"):
                    out, layer_hiddens = _vision_model.forward_intermediates(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                        indices=None,
                    )
                    patch_tokens = _vision_model.post_layernorm(out.last_hidden_state)
                    proj_out = _visual_proj(patch_tokens)
                    return proj_out, layer_hiddens
                else:
                    out = _vision_model(image)
                    patch_tokens = _vision_model.post_layernorm(out.last_hidden_state)
                    proj_out = _visual_proj(patch_tokens)
                    return proj_out, None
                
    elif args.model == "satclip":
        encode_location = model.encode_location
        encode_image = model.encode_image
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    all_locations = load_s2_sample("/home/seri6958/translocator_eval_data/im2gps_places365.csv", "IMG_ID", lon_column="LON", lat_column="LAT")
    all_location_features = encode_locations(all_locations, encode_location, device)
    all_location_features = text_features ## FORCE
    for place in places:
        index_path = os.path.join(place, "index.csv")
        all_locations_place = load_s2_sample(index_path, "id")
    
        if use_surgery:
            out_subdir = (
                "clip_surgery_norm"
                if args.layer_grid_normalize == "minmax"
                else "clip_surgery"
            )
        else:
            out_subdir = "original"
        results_dir = os.path.join(place, out_subdir)
        os.makedirs(results_dir, exist_ok=True)

        # Create a mapping from id to index for O(1) lookup
        id_to_idx = {loc[0]: idx for idx, loc in enumerate(all_locations)}

        # First pass: collect all similarity maps to compute global min/max for consistent color scale
        print(f"Computing {args.model} Surgery similarity maps for {place}...")
        similarity_maps_dict = {}
        rgb_images_dict = {}
        cv2_img_bgr_dict = {}
        
        id_to_layer_hiddens = {}

        for id, lon, lat in tqdm(all_locations_place, desc=f"Computing CLIP Surgery maps for {place}"):
            # if os.path.exists(os.path.join(results_dir, f'{id}.png')):
            #     continue
            data_path = os.path.join(place, data_folder, id)
            try:
                if args.model == "geoclip":
                    image = load_image_geoclip(data_path, device=device)
                    rgb_img = load_rgb_image(data_path)
                    rgb_plot = rgb_img
                else:
                    image, rgb_img = load_image_sentinel(data_path, device=device)
                    rgb_plot = sentinel_rgb_preview_stretch(rgb_img)
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue
            cv2_img_bgr = cv2.cvtColor(rgb_plot, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                image_features, layer_hiddens = encode_image(image)
                id_to_layer_hiddens[id] = layer_hiddens
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                if use_surgery:
                    similarity = clip.clip_feature_surgery(image_features, all_location_features)
                else:
                    similarity = image_features @ all_location_features.t()
                similarity_map = clip.get_similarity_map(similarity[:, 1:, :], rgb_img.shape[:2])
                
                for b in range(similarity_map.shape[0]):
                    for n in range(similarity_map.shape[-1]):
                        target_texts = classes[id]
                        if all_texts[n] not in target_texts:
                            continue
                        
                        vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                        vis = cv2_img_bgr * 0.4 + vis * 0.6
                        vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                        plt.imshow(vis)
                        plt.title(f"CLIP Surgery: {all_texts[n]}")
                        plt.axis("off")
                        plt.savefig(os.path.join(results_dir, f"{id}-{all_texts[n]}.png"), bbox_inches="tight", dpi=150)
                        plt.close()

                        if args.model == "geoclip" and id in id_to_layer_hiddens and id_to_layer_hiddens[id] is not None:
                            heatmaps = []
                            titles = []
                            sum_hm = None
                            with torch.no_grad():
                                for li, hid in enumerate(id_to_layer_hiddens[id][-5:]):
                                    feat = _visual_proj(_vision_model.post_layernorm(hid))
                                    feat = feat / feat.norm(dim=-1, keepdim=True)
                                    sim = clip.clip_feature_surgery(feat, all_location_features)
                                    smap = clip.get_similarity_map(sim[:, 1:, :], rgb_img.shape[:2])
                                    hm = smap[0, :, :, n].cpu().numpy()
                                    heatmaps.append(hm)
                                    titles.append(f"L{li}")
                                    if sum_hm is None:
                                        sum_hm = np.zeros_like(hm, dtype=np.float64)
                                    sum_hm = sum_hm + hm
                                if sum_hm is not None:
                                    smin, smax = float(sum_hm.min()), float(sum_hm.max())
                                    sum_display = (sum_hm - smin) / (smax - smin + 1e-8)
                                    heatmaps.append(sum_display.astype(np.float32))
                                    titles.append("sum")
                            plot_layer_similarity_grid(
                                heatmaps,
                                cv2_img_bgr,
                                os.path.join(results_dir, f"{id}-{all_texts[n]}_layers.png"),
                                titles=titles,
                                suptitle=f"{args.model} per-layer: {all_texts[n]}",
                                ncols=4,
                                heatmap_normalize=args.layer_grid_normalize,
                            )

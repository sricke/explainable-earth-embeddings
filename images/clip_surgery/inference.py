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
import torchvision.transforms as transforms
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
import clip
from images.clip_surgery import get_satclip

BICUBIC = InterpolationMode.BICUBIC


def get_rgb_image(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    image_rgb = np.moveaxis(image.cpu().numpy(), 0, 2)

    image_rgb = image_rgb[:, :, [3, 2, 1]]

    image_rgb = np.clip(image_rgb, 0, 1)  # Ensure [0, 1] range
    image_rgb = (image_rgb * 255).astype("uint8")
    return image_rgb


def load_s2_sample(index_path, id_column):
    df_index = pd.read_csv(index_path)
    # Vectorized operation instead of iterrows (much faster)
    locations = [[id_val, lon, lat] for id_val, lon, lat in zip(df_index[id_column], df_index["lon"], df_index["lat"])]
    return locations


def load_image(path, device="cpu"):
    with rasterio.open(path) as f:
        image = f.read().astype(np.float32)
    image = image / 10000.0
    B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
    image = np.concatenate([image[:10], B10, image[10:]], axis=0)
    image = torch.tensor(image, device=device)
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    return image


def encode_locations(locations: tp.List, model, device):
    # Batch all locations at once instead of one by one
    locations_tensor = torch.tensor([[lon, lat] for _, lon, lat in locations], device=device)
    # Encode all locations in batch
    location_features = model.encode_location(locations_tensor).float()  # (N, #tokens, dim)
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
    
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--model", choices=["satclip", "geoclip"], default="satclip")
    args.add_argument("--arquitecture", choices=["vit16", "resnet50"], default="vit16")
    args.add_argument("--source_folder", type=str, default="/home/seri6958/Places")
    args = args.parse_args()
    
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
            surgery=True,
            return_all=True,
        )  # False for original CLIP
        model.eval()
    elif args.model == "geoclip":
        try:
            geoclip_model_path = SUPPORTED_MODELS["geoclip"][args.arquitecture]
        except KeyError:
            raise ValueError(f"Invalid geoclip arquitecture: {args.arquitecture}")

        model = None # TODO: add geoclip implementation


    places_folder = args.source_folder
    data_folder = "data"
    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]

    for place in places:
        index_path = os.path.join(place, "index.csv")
        all_locations = load_s2_sample(index_path, "id")
        results_dir = os.path.join(place, "clip_surgery_test")
        os.makedirs(results_dir, exist_ok=True)
        all_location_features = encode_locations(all_locations, model, device)

        # Create a mapping from id to index for O(1) lookup
        id_to_idx = {loc[0]: idx for idx, loc in enumerate(all_locations)}

        # First pass: collect all similarity maps to compute global min/max for consistent color scale
        print(f"Computing CLIP Surgery similarity maps for {place}...")
        similarity_maps_dict = {}
        rgb_images_dict = {}
        cv2_img_bgr_dict = {}

        for id, lon, lat in tqdm(all_locations, desc=f"Computing CLIP Surgery maps for {place}"):
            # if os.path.exists(os.path.join(results_dir, f'{id}.png')):
            #     continue
            data_path = os.path.join(place, data_folder, id)
            try:
                image = load_image(data_path, device=device)
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue
            rgb_img = get_rgb_image(image)
            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = clip.clip_feature_surgery(image_features, all_location_features)
                # Use no_norm version to see actual differences (will normalize globally later)
                similarity_map = clip.get_similarity_map(similarity[:, 1:, :], rgb_img.shape[:2])

            n = id_to_idx.get(id)
            if n is None or n >= similarity_map.shape[-1]:
                continue

            similarity_maps_dict[id] = similarity_map[:, :, :, n].cpu()
            rgb_images_dict[id] = rgb_img
            cv2_img_bgr_dict[id] = cv2_img_bgr

        # Compute global min/max across all similarity maps for consistent color scale
        if similarity_maps_dict:
            all_values = torch.cat([sm.flatten() for sm in similarity_maps_dict.values()])
            global_min = all_values.min().item()
            global_max = all_values.max().item()
            print(f"Global similarity range: [{global_min:.4f}, {global_max:.4f}]")
        else:
            global_min, global_max = 0.0, 1.0

        # Second pass: normalize and visualize with consistent color scale
        for id in tqdm(similarity_maps_dict.keys(), desc=f"Visualizing CLIP Surgery maps for {place}"):
            similarity_map = similarity_maps_dict[id]
            cv2_img_bgr = cv2_img_bgr_dict[id]

            # Normalize using global min/max for consistent color scale
            if global_max > global_min:
                similarity_map_norm = (similarity_map - global_min) / (global_max - global_min)
            else:
                similarity_map_norm = similarity_map

            for b in range(similarity_map_norm.shape[0]):
                vis = (similarity_map_norm[b, :, :].numpy() * 255).astype("uint8")
                # Apply colormap to convert 2D grayscale to 3-channel BGR
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

                # Blend: 80% background image, 20% similarity map
                vis = cv2_img_bgr * 0.8 + vis * 0.2
                vis = np.clip(vis, 0, 255).astype("uint8")

                # Convert BGR to RGB for matplotlib display
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                plt.imshow(vis)
                plt.title("CLIP Surgery")
                plt.axis("off")
                plt.savefig(os.path.join(results_dir, f"{id}.png"), bbox_inches="tight", dpi=150)
                plt.close()  # Free memory

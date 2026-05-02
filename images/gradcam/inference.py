import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "CLIP_Surgery") not in sys.path:
    sys.path.insert(0, str(_root / "CLIP_Surgery"))
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import rasterio
import torch.nn.functional as F

from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from images.clip_surgery import get_satclip
from images.clip_surgery.geoclip_surgery.load import get_geoclip

BICUBIC = InterpolationMode.BICUBIC

preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])

_geoclip_preprocess = Compose([
    Resize((224, 224), interpolation=BICUBIC),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])


def load_rgb_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image).astype("float32") / 255.0


def get_rgb_image_sentinel(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    image_rgb = np.moveaxis(image.cpu().numpy(), 0, 2)
    image_rgb = image_rgb[:, :, [3, 2, 1]]
    return np.clip(image_rgb, 0, 1).astype("float32")


def load_s2_sample(index_path, id_column, lon_column="lon", lat_column="lat"):
    df_index = pd.read_csv(index_path)
    locations = [[id_val, lon, lat] for id_val, lon, lat in zip(df_index[id_column], df_index[lon_column], df_index[lat_column])]
    return locations


def load_image_geoclip(path, device="cpu"):
    image = Image.open(path).convert("RGB")
    return _geoclip_preprocess(image).unsqueeze(0).to(device)


def load_image_sentinel(path, device="cpu"):
    with rasterio.open(path) as f:
        image = f.read().astype(np.float32)
    image = image / 10000.0
    B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
    image = np.concatenate([image[:10], B10, image[10:]], axis=0)
    image = torch.tensor(image, device=device)
    return preprocess(image.unsqueeze(0))


class EmbeddingOutputTarget:
    def __init__(self, target):
        self.target = target if isinstance(target, torch.Tensor) else torch.tensor(target)

    def __call__(self, model_output):
        self.target = self.target.to(model_output.device)
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        norm_output = model_output / model_output.norm(dim=1, keepdim=True)
        norm_target = self.target / self.target.norm(dim=1, keepdim=True)
        return -F.cosine_similarity(norm_output, norm_target, dim=-1)
    
def vit_reshape_transform(tensor, height=16, width=16):
  # CLIP ViT token output: [B, 1 + H*W, C]
  tensor = tensor[:, 1:, :]  # remove CLS token
  tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
  tensor = tensor.permute(0, 3, 1, 2)  # [B, C, H, W]
  return tensor


class GeoCLIPVisualWrapper(torch.nn.Module):
    """Wraps GeoCLIP image encoder into a single module for GradCAM."""
    def __init__(self, vision_model, visual_proj, mlp):
        super().__init__()
        self.vision_model = vision_model
        self.visual_proj = visual_proj
        self.mlp = mlp

    def forward(self, pixel_values):
        out = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=False)
        proj_out = self.visual_proj(out.pooler_output)
        return self.mlp(proj_out)


SUPPORTED_MODELS = {
    "satclip": {
        "vit16": "microsoft/SatCLIP-ViT16-L40",
    },
}


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("--model", choices=["satclip", "geoclip"], default="satclip")
    args.add_argument("--arquitecture", choices=["vit16"], default="vit16")
    args.add_argument("--source_folder", type=str, default="/home/seri6958/Places")
    args = args.parse_args()

    if args.model == "satclip":
        try:
            satclip_model_path = SUPPORTED_MODELS["satclip"][args.arquitecture]
        except KeyError:
            raise ValueError(f"Invalid satclip arquitecture: {args.arquitecture}")

        model = get_satclip(
            hf_hub_download(satclip_model_path, "satclip-vit16-l40.ckpt"),
            device=device,
            return_all=True,
        )
        model.eval()
        encode_location = model.encode_location
        visual_model = model.visual
        target_layers = [model.visual.blocks[-1]]

    elif args.model == "geoclip":
        geo_model = get_geoclip(device=device, return_all=True)
        encode_location = geo_model.location_encoder.forward
        visual_model = GeoCLIPVisualWrapper(
            geo_model.image_encoder.CLIP.vision_model,
            geo_model.image_encoder.CLIP.visual_projection,
            geo_model.image_encoder.mlp,
        )
        target_layers = [geo_model.image_encoder.CLIP.vision_model.encoder.layers[-1].layer_norm1]
    else:
        raise ValueError(f"Invalid model: {args.model}")

    places_folder = args.source_folder
    data_folder = "data" if args.model == "satclip" else "images"
    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]

    for place in places:
        index_path = os.path.join(place, "index.csv")
        all_locations_place = load_s2_sample(index_path, "id")
        results_dir = os.path.join(place, "gradcam")
        os.makedirs(results_dir, exist_ok=True)

        print(f"Computing {args.model} GradCAM maps for {place}...")

        for id, lon, lat in tqdm(all_locations_place, desc=f"Computing GradCAM for {place}"):
            data_path = os.path.join(place, data_folder, id)
            try:
                if args.model == "geoclip":
                    image = load_image_geoclip(data_path, device=device)
                else:
                    image = load_image_sentinel(data_path, device=device)
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue

            if args.model == "geoclip":
                rgb_img = load_rgb_image(data_path)
            else:
                rgb_img = get_rgb_image_sentinel(image)

            image = image.requires_grad_(True)

            with torch.no_grad():
                loc_tensor = torch.tensor([[lon, lat]], device=device)
                loc_feat = encode_location(loc_tensor).float()
                loc_feat = loc_feat / loc_feat.norm(dim=-1, keepdim=True)
                if loc_feat.dim() == 3:
                    loc_feat = loc_feat.squeeze(1)

            targets = [EmbeddingOutputTarget(loc_feat)]

            with GradCAM(
                model=visual_model,
                target_layers=target_layers,
                reshape_transform=vit_reshape_transform,
            ) as cam:
                grayscale_cam = cam(input_tensor=image, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

            h, w = rgb_img.shape[:2]
            if grayscale_cam.shape != (h, w):
                grayscale_cam = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)

            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            plt.imshow(visualization)
            plt.title(f"GradCAM ({args.model})")
            plt.axis("off")
            plt.savefig(os.path.join(results_dir, f"{id}.png"), bbox_inches="tight", dpi=150)
            plt.close()

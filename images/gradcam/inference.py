#!/usr/bin/env python3
"""
Visualization script for SatCLIP with GradCAM
Converted from visualize.ipynb
"""

import sys
import os
from pathlib import Path

# Add project root to path so we can import satclip
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Add satclip/satclip directory to path for relative imports within satclip module
satclip_dir = project_root / "satclip" / "satclip"
if str(satclip_dir) not in sys.path:
    sys.path.insert(0, str(satclip_dir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import rasterio  # Better for multi-band satellite imagery
import torch
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from satclip.satclip.load import get_satclip
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def get_rgb_image(
        image
    ):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image_rgb = np.rollaxis(image.numpy(), 0, 3)
        
        image_rgb = (image_rgb[:, :, [3,2,1]])
        
        image_rgb = np.clip(image_rgb, 0, 1).astype('float32')  # Ensure [0, 1] range
        #image_rgb = (image_rgb * 255).astype('float32')
        return image_rgb
    
def load_s2_sample(index_path, target_biome):
    df_index = pd.read_csv(index_path)
    locations = []
    target_locations = []
    for index, row in df_index.iterrows():
        locations.append([row['lon'], row['lat']])
        if pd.isna(row['biome']) or row['biome'] == "":
            target_locations.append((row['fn'], [row['lon'], row['lat']]))
    return locations, target_locations

def load_image(path):
    with rasterio.open(path) as f:
        image = f.read().astype(np.float32)
    image = image/10000.0
    B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
    image = np.concatenate([image[:10], B10, image[10:]], axis=0)
    image = torch.tensor(image)
    image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    return image


class EmbeddingOutputTarget:
    def __init__(self, target):
        if isinstance(target, torch.Tensor):
            self.target = target
        else:
            self.target = torch.tensor(target)
            
    def __call__(self, model_output):
        self.target = self.target.to(model_output.device)
        
        if len(model_output.shape) == 1:
            model_output = model_output.unsqueeze(0)
        
        # Normalize to unit vectors
        norm_model_output = model_output / model_output.norm(dim=1, keepdim=True)
        norm_target = self.target / self.target.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity
        return - torch.nn.functional.cosine_similarity(norm_model_output, norm_target, dim=-1)
    
def main():
    # Configuration
    image_id = "patch_2290.tif"
    image_folder = "images1"
    image_path = os.path.join(image_folder, image_id)
    results_folder = "results_cam"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    # Load index CSV
    print("Loading index CSV...")
    
    locations, target_locations = load_s2_sample("sample_index.csv", None)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    satclip_model_path = "microsoft/SatCLIP-ViT16-L40"
    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ResNet50-L10", "satclip-resnet50-l10.ckpt"),
        device=device,
        return_all=True, # to also return the image model
    )  
    
    model.eval()
    image_folder = "/home/seri6958/explainable-earth-embeddings/satclip/figures/ssl4eo/example/images"
    for image_id, location in target_locations:
        image = load_image(os.path.join(image_folder, image_id))
        # Move image to device and enable gradients for GradCAM
        image = image.to(device).requires_grad_(True)
        
        image_features = model.encode_image(image)
        location = torch.tensor([location], device=device)
        location_features = model.encode_location(location).float()
        print('location_features mean: ', location_features.mean().item())
        print('image_features mean: ', image_features.mean().item())
        # Setup GradCAM
        #target_layers = [model.visual.blocks[-1]]
        target_layers = [model.visual.layer4[-1]]
        targets = [EmbeddingOutputTarget(location_features)]
    
        rgb_img = get_rgb_image(image.detach().cpu())
        save_dir = os.path.join(results_folder, image_id.replace('.tif', ''))
        os.makedirs(save_dir, exist_ok=True)
        plt.imshow(rgb_img)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, 'rgb.png'), bbox_inches='tight', dpi=150)
        # Run GradCAM
        print("Running GradCAM...")
        with GradCAM(model=model.visual, target_layers=target_layers) as cam:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=image, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            model_outputs = cam.outputs
        # Save visualization
        plt.imshow(visualization)
        plt.title("GradCAM")
        plt.axis('off')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'gradcam.png'), bbox_inches='tight', dpi=150)
        print(f"Saved GradCAM visualization to '{os.path.join(save_dir, 'gradcam.png')}'")
        plt.close()
        

if __name__ == "__main__":
    main()
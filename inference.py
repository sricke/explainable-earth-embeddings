import sys
import os
import json
from pathlib import Path
parent = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(parent))
sys.path.append(str(parent.parent / "SpLiCE"))
from splice import *
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pandas as pd
from splice import SPLICE
from tqdm import tqdm
import open_clip
from main import Location2TextLightningModule
from torch.utils.data import DataLoader

def get_example_image(path, plot=True, results_folder="results"):
    """
    Loads example Sentinel-2 image from path to tif file
    :param path: Path to image file
    :param plot: If True, will plot an RGB image of the satellite image
    :return: Normalized image as torch tensor
    """
    os.makedirs(results_folder, exist_ok=True)
    with rasterio.open(path) as f:
        data = f.read().astype(np.float32)
        image = data / 10000.0
        B10 = np.zeros((1, *image.shape[1:]), dtype=image.dtype)
        image = np.concatenate([image[:10], B10, image[10:]], axis=0)
        image = torch.tensor(image)

        if plot:
            # Read the first three RGB bands (assuming they are bands 4, 3, and 2)
            red_band = f.read(4) / 10000.0
            green_band = f.read(3) / 10000.0
            blue_band = f.read(2) / 10000.0

            # Stack the bands to create an RGB image
            rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)
            
            # Resize image from 224x224x3 to 256x256x3
            rgb_image = (rgb_image * 255).astype(np.uint8)  # Convert to uint8 for PIL
            rgb_image = Image.fromarray(rgb_image)
            rgb_image = rgb_image.resize((224, 224), Image.Resampling.LANCZOS)
            rgb_image = np.array(rgb_image).astype(np.float32) / 255.0  # Convert back to float32
            
            # Apply contrast stretching for better visualization (percentile stretching)
            # Clip values to 2nd and 98th percentiles and normalize to 0-1
            p2, p98 = np.percentile(rgb_image, (2, 98))
            rgb_image_display = np.clip((rgb_image - p2) / (p98 - p2), 0, 1)

            # Plot the RGB image
            plt.imshow(rgb_image)
            plt.title("RGB Sentinel-2 Image")
            plt.axis('off')  # Hide axes
            if plot:
                plt.savefig(f'{results_folder}/original_rgb_image_{path.split("/")[-1]}.png', bbox_inches='tight', dpi=150)
            plt.close()

    return rgb_image_display

# Load Lightning checkpoint
checkpoint_path = "/home/ricke/satcam/satcam/checkpoints/Location2Text_S2-epoch=99-val_loss=0.1697.ckpt"

model = Location2TextLightningModule.load_from_checkpoint(checkpoint_path, device="cuda")
model.eval()

vocab_path = str(parent / 'satellite_vocab_refined.json')
## Compute embedded dictionary, mean-center and normalize
use_satellite_concepts = True
if use_satellite_concepts:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    # not considering plural words
    # vocab = [v for v in vocab if not vocab[v]['is_plural']]
else:
    vocab = []
    vocabulary_size = 10000
    with open("/home/ricke/.cache/splice/vocab/laion.txt", "r") as f:
        lines = f.readlines()
        if vocabulary_size > 0:
            lines = lines[-vocabulary_size:]
        for line in lines:
            vocab.append(line.strip())
            
satellite_concepts = []
satellite_concepts_words = []
for idx, word in tqdm(enumerate(vocab), total=len(vocab), desc="Encoding satellite concepts"):
    concept = model.text_model_predict(word)
    # Squeeze to remove batch dimension if present
    if concept.dim() > 1:
        concept = concept.squeeze(0)
    satellite_concepts.append(concept)
    satellite_concepts_words.append(word)
satellite_concepts = torch.stack(satellite_concepts).to("cuda")

mean_embedding_concepts = torch.mean(satellite_concepts, dim=0).to("cuda")
satellite_concepts = torch.nn.functional.normalize(satellite_concepts, dim=1)
satellite_concepts = torch.nn.functional.normalize(satellite_concepts-mean_embedding_concepts, dim=1)

satclip_df = pd.read_csv(str(parent / "satclip_index.csv"))
satclip_embeddings = torch.zeros(len(satclip_df), 512).to("cuda")
with torch.no_grad():
    locations = satclip_df[["lat", "lon"]].values.astype(np.float64)
    locations = torch.tensor(locations).to("cuda")
    dataloader = DataLoader(locations, batch_size=6960, shuffle=False)
    i = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding location embeddings"):
        satclip_embeddings[i:i+batch.shape[0]] = model.location_model(batch)
        i += batch.shape[0]
        

    mean_embedding = torch.mean(satclip_embeddings, dim=0).to("cuda")
    #mean_embedding = torch.load("/home/ricke/.cache/splice/means/open_clip_ViT-B-32_image.pt").to("cuda")
    
    splice = SPLICE(mean_embedding, satellite_concepts, clip=model.text_model, l1_penalty=0.01, device="cuda")

    # Initialize list to store results
    results = []
    
    for index, location_embedding in tqdm(enumerate(satclip_embeddings), total=len(satclip_embeddings), desc="Decomposing location embeddings"):
        location_embedding = location_embedding.unsqueeze(0).to("cuda")  # Add batch dimension and move to CUDA
        
        location_embedding = torch.nn.functional.normalize(location_embedding, dim=1)
        centered_location_embedding = torch.nn.functional.normalize(location_embedding-mean_embedding, dim=1)
        
        sparse_weights = splice.decompose(centered_location_embedding.detach())
        
        recon_location = sparse_weights@splice.dictionary
        recon_location = torch.nn.functional.normalize(recon_location, dim=1)
        recon_location = torch.nn.functional.normalize(recon_location + mean_embedding, dim=1)

        sparse_weights = sparse_weights.squeeze()
        
        cosine_similarity = torch.nn.functional.cosine_similarity(recon_location.double(), location_embedding.double(), dim=1)
        cosine = torch.diag(recon_location @ location_embedding.T).sum()
        print(f"Cosine similarity: {cosine_similarity.item()}, cosine: {cosine.item()}")
        location = satclip_df.iloc[index]
        image_path = location['fn']
        
        print(f"Location: {image_path} {location['lat']}, {location['lon']}")
        
        image_path = os.path.join("/home/ricke/satcam/images1", image_path)
        if not os.path.exists(image_path):
            continue
        rgb_image = get_example_image(image_path)
        
        sparse_weights = sparse_weights.squeeze()
        sorted_weights = torch.sort(sparse_weights, descending=True)[1][:10]
        for weight_idx in sorted_weights:
            print(f"{satellite_concepts_words[weight_idx.item()]}: {sparse_weights[weight_idx].item()}")
        breakpoint()
        # Store weights for this location
        weights_array = sparse_weights.detach().cpu().numpy()
        results.append({
            'lat': location['lat'],
            'lon': location['lon'],
            'fn': location['fn'],
            'weights': weights_array,
            'cosine_similarity': cosine_similarity.item()
        })
        
        breakpoint()
    
    # Create DataFrame and save to parquet
    weights_df = pd.DataFrame(results)
    output_path = str(parent / "location_weights.parquet")
    weights_df.to_parquet(output_path, index=False)
    print(f"Saved weights for {len(results)} locations to {output_path}")
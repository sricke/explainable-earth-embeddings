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
checkpoint_path = "/home/ricke/satcam/satcam/checkpoints-alignment-encoder-10/Location2Text_S2-epoch=11-val_loss=12.2384.ckpt"

model = Location2TextLightningModule.load_from_checkpoint(checkpoint_path, device="cuda")
model.eval()

vocab_path = str(parent / 'satellite_vocab_land.json')
## Compute embedded dictionary, mean-center and normalize
use_satellite_concepts = True
satellite_concepts_words = []
satellite_concepts = []
if use_satellite_concepts:
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    for word in vocab:
        embedding = model.text_model_predict(word, normalize=True)
        # Ensure 1D tensor [embed_dim] - single word returns [1, embed_dim]
        if len(embedding.shape) == 2:
            embedding = embedding.squeeze(0)  # Remove batch dimension
        satellite_concepts.append(embedding)
        satellite_concepts_words.append(word)
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
            line = line.strip()
            vocab.append(line)
            embedding = model.text_model_predict(line, normalize=True) #normalize text
            # Ensure 1D tensor [embed_dim] - single word returns [1, embed_dim]
            if len(embedding.shape) == 2:
                embedding = embedding.squeeze(0)  # Remove batch dimension
            satellite_concepts.append(embedding)
            satellite_concepts_words.append(line)

satellite_concepts = torch.stack(satellite_concepts)  # Should be [num_concepts, embed_dim]
# Verify it's 2D
if len(satellite_concepts.shape) != 2:
    raise ValueError(f"Expected 2D tensor [num_concepts, embed_dim], got shape {satellite_concepts.shape}")


satellite_concepts = torch.nn.functional.normalize(satellite_concepts, dim=1) # this normalize is redundant because it is already normalized in the model
satellite_concepts = torch.nn.functional.normalize(satellite_concepts-torch.mean(satellite_concepts, dim=0), dim=1) 


satclip_df = pd.read_csv(str(parent / "satclip_index.csv"))
satclip_embeddings = torch.zeros(len(satclip_df), 512).to("cuda")
with torch.no_grad():
    locations = satclip_df[["lon", "lat"]].values.astype(np.float64)
    locations = torch.tensor(locations).to("cuda")
    dataloader = DataLoader(locations, batch_size=6960, shuffle=False)
    i = 0
    for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding location embeddings"):
        satclip_embeddings[i:i+batch.shape[0]] = model.location_model(batch) # unnormalized
        i += batch.shape[0]

    satclip_embeddings = torch.nn.functional.normalize(satclip_embeddings, dim=1) 
    mean_embedding_locs = torch.mean(satclip_embeddings, dim=0).to("cuda") #get mean
    
    splicemodel = SPLICE(mean_embedding_locs, satellite_concepts, clip=model, device="cuda", return_weights=True, return_cosine=True, l1_penalty=0.01)
    
   
    for batch_index, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Going through batches"):
        # Process each location coordinate in the batch
        results = []
        for index, location_coords in tqdm(enumerate(batch), total=len(batch), desc="Decomposing location embeddings"):
            # Ensure 2D tensor [1, 2] for encode_image
            location_coords = location_coords.unsqueeze(0)  # shape: [1, 2]
            sparse_weights, cosine_similarity = splicemodel.encode_image(location_coords)         # shape = [1, 10000], l0 norm = 9
            reconstruction = splicemodel.recompose_image(sparse_weights)   # shape = [1, 512]  
            sparse_weights = sparse_weights.squeeze()
            
            image_index = batch_index * 6960 + index
            location = satclip_df.iloc[image_index]
            image_path = location['fn']
            """
            image_path = os.path.join("/home/ricke/satcam/images1", image_path)
            if not os.path.exists(image_path):
                continue
            print(image_path)
            rgb_image = get_example_image(image_path)
            """
            """ print(f"Cosine similarity: {cosine_similarity.item()}")
            for i, weight_idx in enumerate(torch.sort(sparse_weights, descending=True)[1]):

                print(f"{satellite_concepts_words[weight_idx.item()]}: {sparse_weights[weight_idx.item()].item()}")"""
            # Store weights for this location
            weights_array = sparse_weights.detach().cpu().numpy()
            result = {}
            result['lat'] = location['lat']
            result['lon'] = location['lon']
            result['fn'] = location['fn']
            result['weights'] = weights_array
            result['cosine_similarity'] = cosine_similarity.item()
            for index in range(len(sparse_weights)):
                result[satellite_concepts_words[index]] = sparse_weights[index].item()
            results.append(result)
        # Create DataFrame and save to parquet
        weights_df = pd.DataFrame(results)
        output_path = str(parent / f"location_weights_batch_{batch_index}.parquet")
        weights_df.to_parquet(output_path, index=False)
        print(f"Saved weights for {len(results)} locations to {output_path}")





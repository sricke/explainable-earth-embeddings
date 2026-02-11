import sys
import os
import json
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

if __name__ == "__main__":
    from IPython import embed; embed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Lightning checkpoint
    checkpoint_path = "../../outputs/explainable-earth-embeddings/checkpoints/Text2Location-epoch=199-val_loss=1.6525.ckpt"

    model = Location2TextLightningModule.load_from_checkpoint(checkpoint_path, device="cuda")
    model.eval()

    vocab_path = str(parent / 'satellite_vocab_land.json')
    ## Compute embedded dictionary, mean-center and normalize

    satellite_concepts_words = []
    satellite_concepts = []
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    for word in tqdm(vocab, desc="Iterating through vocab..."):
        embedding = model.text_model_predict(word, normalize=True)
        if len(embedding.shape) == 2:
            embedding = embedding.squeeze(0)
        satellite_concepts.append(embedding.cpu())  # Move to CPU immediately
        satellite_concepts_words.append(word)
    
    # Clear GPU memory after encoding vocab
    torch.cuda.empty_cache()

    satellite_concepts = torch.stack(satellite_concepts).to(device)
    satellite_concepts = torch.nn.functional.normalize(satellite_concepts, dim=1)
    satellite_concepts = torch.nn.functional.normalize(satellite_concepts-torch.mean(satellite_concepts, dim=0), dim=1)

    satclip_df = pd.read_csv(str(parent / "text-descriptions/index.csv"))

    output_dim = model.output_dim
    satclip_embeddings = torch.zeros(len(satclip_df), output_dim).to(device)

    with torch.no_grad():
        locations = satclip_df[["lon", "lat"]].values.astype(np.float64)
        locations = torch.tensor(locations, device=device)  # Create directly on device
        
        # Reduce batch size to prevent memory issues
        dataloader = DataLoader(locations, batch_size=512, shuffle=False)  # Reduced from 6960
        i = 0
        for batch in tqdm(dataloader, total=len(dataloader), desc="Encoding location embeddings"):
            satclip_embeddings[i:i+batch.shape[0]] = model.location_model(batch)
            i += batch.shape[0]
            
            # Clear cache every N batches
            if i % 5120 == 0:  # Every 10 batches with batch_size=512
                torch.cuda.empty_cache()

    satclip_embeddings = torch.nn.functional.normalize(satclip_embeddings, dim=1) 
    mean_embedding_locs = torch.mean(satclip_embeddings, dim=0).to(device)

    l1_penalty = 0.05
    splicemodel = SPLICE(
        mean_embedding_locs, 
        satellite_concepts, 
        clip=None,  # <-- set to None since we're passing pre-computed embeddings
        device="cuda", 
        return_weights=True, 
        return_cosine=True, 
        l1_penalty=l1_penalty # might want to make smaller....
    )

    # iterate through PRECOMPUTED embeddings
    results = []
    for index in tqdm(range(len(satclip_embeddings)), desc="Decomposing location embeddings"):
        location_embedding = satclip_embeddings[index].unsqueeze(0)  # shape: [1, 256]
        
        # normalize and center
        location_embedding = torch.nn.functional.normalize(location_embedding, dim=1)
        centered_location_embedding = torch.nn.functional.normalize(location_embedding - mean_embedding_locs, dim=1)
        
        # decompose using pre-computed embedding
        sparse_weights = splicemodel.decompose(centered_location_embedding)
        
        # reconstruct and compute similarity
        recon_location = splicemodel.recompose_image(sparse_weights)
        recon_location = torch.nn.functional.normalize(recon_location, dim=1)
        recon_location = torch.nn.functional.normalize(recon_location + mean_embedding_locs, dim=1)
        
        cosine_similarity = torch.nn.functional.cosine_similarity(
            recon_location, location_embedding, dim=1
        )
        
        sparse_weights = sparse_weights.squeeze()
        
        location = satclip_df.iloc[index]
        
        result = {
            'lat': location['lat'],
            'lon': location['lon'],
            'fn': location['fn'],
            'cosine_similarity': cosine_similarity.item()
        }
        
        for concept_idx, concept_word in enumerate(satellite_concepts_words):
            result[concept_word] = sparse_weights[concept_idx].item()
        
        results.append(result)
        
        if (index + 1) % 10000 == 0:
            batch_num = (index + 1) // 10000
            weights_df = pd.DataFrame(results)
            output_path = str(parent / f"location_weights_batch_{batch_num}.parquet")
            weights_df.to_parquet(output_path, index=False)
            print(f"Saved batch {batch_num} with {len(results)} locations to {output_path}")
            results = [] 
            torch.cuda.empty_cache()

    if results:
        from IPython import embed; embed()
        batch_num = (len(satclip_embeddings) // 10000) + 1
        weights_df = pd.DataFrame(results)
        output_path = str(parent / f"location_weights_batch_{batch_num}_l1_penalty_{l1_penalty}_num_samples_{num_samples}.parquet")
        weights_df.to_parquet(output_path, index=False)
        print(f"Saved final batch with {len(results)} locations to {output_path}")
        torch.cuda.empty_cache()
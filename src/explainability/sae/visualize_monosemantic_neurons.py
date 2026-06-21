import os
import torch
import pandas as pd
import numpy as np
import rasterio
import shutil
from PIL import Image
from tqdm import tqdm

def save_top_neuron_images(
    scores_path="all_neurons_scores.pth",
    activations_path="sparse_activations.csv",
    output_dir="top_neurons_visuals",
    top_k_neurons=10,
    images_per_neuron=10,
    largest=True
):
    
    scores = torch.load(scores_path, map_location="cpu")

    if largest == True:
        output_dir = os.path.join(output_dir, "highest")
        scores[torch.isnan(scores)] = -1e9
    else:        
        output_dir = os.path.join(output_dir, "lowest")
    
    top_scores, top_neuron_indices = torch.topk(scores, top_k_neurons, largest=largest)

    print("Loading dataframes...")
    activations_df = pd.read_csv(activations_path)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing top {top_k_neurons} neurons...")
    for rank, neuron_idx in enumerate(top_neuron_indices):
        n_idx = neuron_idx.item()
        print(f"Rank {rank+1}: Neuron {n_idx} with score {top_scores[rank].item():.4f}")
        
        column_name = f"act{n_idx + 1}"

        top_samples = activations_df.nlargest(images_per_neuron, column_name)
        
        neuron_dir = os.path.join(output_dir, f"rank_{rank+1}_neuron_{n_idx}")
        os.makedirs(neuron_dir, exist_ok=True)

        for i, (_, sample) in enumerate(top_samples.iterrows()):
            
            img_path = sample['filename']
            image_basename = os.path.basename(img_path)
            tif_save_path = os.path.join(neuron_dir, f"top_{i+1}_orig_{image_basename}")
            shutil.copy2(img_path, tif_save_path)

            print(sample[column_name])
            if img_path.lower().endswith(('.tif', '.tiff')):
                try:
                    with rasterio.open(img_path) as src:
                        rgb = src.read([4, 3, 2]).astype(np.float32)
                        p2, p98 = np.percentile(rgb, (2, 98))

                        if p98 - p2 > 0:
                            rgb_normalized = (rgb - p2) / (p98 - p2)
                        else:
                            rgb_normalized = rgb - p2

                        rgb_normalized = np.clip(rgb_normalized, 0, 1)

                        rgb_8bit = (rgb_normalized * 255).astype(np.uint8)
                        
                        # Transpose to (H, W, C)
                        img_array = np.transpose(rgb_8bit, (1, 2, 0))
                        img = Image.fromarray(img_array)
                        
                        save_path = os.path.join(neuron_dir, f"top_{i+1}_act_{image_basename}.png")
                        img.save(save_path)
                        
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    print(f"Visual examples for monosemantic neurons saved in {output_dir}")

if __name__ == "__main__":
    base_dir = "/home/results/sae_geo_embeddings/S2100k/climplicit/BatchTopKTrainer/dict_size=1024,k=20/20260602_210907/xAI/"
    monosemanticity_type = "visual_monosemanticity"
    monosemanticity_dir = os.path.join(base_dir, monosemanticity_type)
    monosemanticity_scores_path = os.path.join(monosemanticity_dir, "all_neurons_scores.pth")
    output_dir = os.path.join(monosemanticity_dir, "top_neurons_visuals")
    activations_csv_path = os.path.join(base_dir, "sparse_activations.csv")
    save_top_neuron_images(
        scores_path=monosemanticity_scores_path,
        activations_path=activations_csv_path,
        output_dir=output_dir,
        largest=True)
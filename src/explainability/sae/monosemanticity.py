import torch
import os.path
import argparse
import os

import tqdm
import torch.nn.functional as F
import pandas as pd
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#device = "cpu"

def get_args_parser():
    parser = argparse.ArgumentParser("Measure monosemanticity via weighted pairwise cosine similarity", add_help=False)
    parser.add_argument("--embeddings_path")
    parser.add_argument("--activations_path")
    return parser

def haversine_distance(coords1, coords2):
    """
    Calculates the Haversine distance between two sets of points.
    coords1: (1, 2) tensor of [lat, lon] in radians
    coords2: (batch, 2) tensor of [lat, lon] in radians
    Returns: Distance in kilometers
    """
    R = 6371.0  # Earth radius in km
    
    lat1, lon1 = coords1[:, 0], coords1[:, 1]
    lat2, lon2 = coords2[:, 0], coords2[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c

def calc_monosemanticity(geo_coordinates, embeddings, activations, type="visual", batch_size=200):
    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=device)
    weight_sum = torch.zeros(num_neurons, device=device)
    batch_size = 50000  # Set batch size
    #num_images=3000

    for i in tqdm.tqdm(range(num_images), desc="Processing image pairs"):
        for j_start in range(i + 1, num_images, batch_size):  # Process in batches
            j_end = min(j_start + batch_size, num_images)
            
            activations_i = activations[i].cuda()  # (num_neurons)
            activations_j = activations[j_start:j_end].cuda()  # (batch_size, num_neurons)

            # Compute cosine similarity
            if type == "visual":
                embeddings_i = embeddings[i].cuda()  # (embedding_dim)
                embeddings_j = embeddings[j_start:j_end].cuda()  # (batch_size, embedding_dim)
                cosine_similarities = F.cosine_similarity(
                    embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),  # Expanding to (batch_size, embedding_dim)
                    embeddings_j,
                    dim=1)
            else:
                coords_i = geo_coordinates[i].unsqueeze(0) # (1, 2)
                coords_j = geo_coordinates[j_start:j_end]  # (batch, 2)
                dist = haversine_distance(coords_i, coords_j)
                cosine_similarities = torch.exp(-dist / 500.0)

            # Compute weights and weighted similarities
            # Expanding activations_i to (1, num_neurons)
            weights = activations_i.unsqueeze(0) * activations_j  # (batch_size, num_neurons)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch_size, num_neurons)

            weighted_cosine_similarities = torch.sum(weighted_cosine_similarities, dim=0)  # (num_neurons)
            weighted_cosine_similarity_sum += weighted_cosine_similarities

            weights = torch.sum(weights, dim=0)  # (num_neurons)
            weight_sum += weights

    monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan)

    results_dir = os.path.join(os.path.dirname(args.activations_path), "{}_monosemanticity".format(type))
    os.makedirs(results_dir, exist_ok=True)
    print(monosemanticity)
    torch.save(monosemanticity, os.path.join(results_dir, "all_neurons_scores.pth"))

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])

    print(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}")
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, 10, largest=False)

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    # Save to file
    output_path = os.path.join(results_dir, "metric_stats_new.txt")
    with open(output_path, "w") as file:
        file.write(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()}\n")
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")

        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")

        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")
    
    return monosemanticity.detach().cpu().flatten()

def main(args):
    # Load embeddings
    embeddings = torch.load(args.embeddings_path, map_location=device)
    print(f"Loaded embeddings found at {args.embeddings_path}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Load activations
    df = pd.read_csv(args.activations_path)

    geo_coordinates = torch.tensor(np.radians(df[['lat', 'lon']].values), dtype=torch.float32).to(device)

    activations_df = df.drop(columns=["filename",'lat', 'lon'])
    
    sparse_cols_mask = (activations_df != 0).sum() < 20
    
    activations_df.loc[:, sparse_cols_mask] = 0.0

    activations_tensor = torch.tensor(activations_df.values, dtype=torch.float32)
    print(f"Loaded activations found at {args.activations_path}")
    print(f"Activations shape: {activations_tensor.shape}")

    min_values = activations_tensor.min(dim=0, keepdim=True)[0]
    max_values = activations_tensor.max(dim=0, keepdim=True)[0]
    activations = (activations_tensor - min_values) / (max_values - min_values)

    visual_neuron_ms = calc_monosemanticity(geo_coordinates, embeddings, activations, type="visual")
    geospatial_neuron_ms = calc_monosemanticity(geo_coordinates, embeddings, activations, type="geospatial")

    combined_ms_scores = pd.DataFrame({
    'Visual MS': pd.Series(visual_neuron_ms),
    'Geospatial MS': pd.Series(geospatial_neuron_ms)})

    output_file = os.path.join(os.path.dirname(args.activations_path), "join_monosemanticity.csv")
    combined_ms_scores.to_csv(output_file)



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
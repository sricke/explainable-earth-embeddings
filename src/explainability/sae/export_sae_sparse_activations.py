#!/usr/bin/env python3
import argparse
import csv
import os

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_download


from saes import BatchTopKSAE

from satclip.satclip.load import get_satclip
from satclip.satclip.datamodules.s2geo_dataset import S2GeoDataModule

from training import get_norm_factor
from util import *

device = "cuda" if t.cuda.is_available() else "cpu"

try:
    from geoclip import LocationEncoder, ImageEncoder
except ImportError as exc:
    raise ImportError(
        "Could not import LocationEncoder from geoclip. "
        "Install the geoclip package or activate the correct Python environment."
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export sparse SAE activations for all points in an S2Geo dataset."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root folder of the S2Geo dataset containing index.csv and sen2_images/.",
    )
    parser.add_argument(
        "--location_encoder",
        default="satclip",
        help="The class of the location encoder",
    )
    parser.add_argument(
        "--sae-path",
        required=True,
        help="Path to the pretrained SAE checkpoint file (.pt) to load with from_pretrained().",
    )
    return parser.parse_args()

def build_csv_header(dict_size):
    return ["filename","lon", "lat"] + [f"act{i+1}" for i in range(dict_size)]


def export_activations(data_dir, sae_path, location_encoder):
    sae_model_path = os.path.join(sae_path, "best_ae.pt")
    output_csv = os.path.join(sae_path, "xAI/" "sparse_activations.csv")
    print(f"Output CSV will be saved to: {output_csv}")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    visual_embeddings_path = os.path.join(sae_path, "xAI/visual_embeddings.pt")

    print(f"Using device: {device}")
    print(f"Loading pretrained SAE from: {sae_model_path}")
    autoencoder = BatchTopKSAE.from_pretrained(sae_model_path, device=device).double()
    autoencoder.eval()

    print("Initializing geo and image encoder...")
    location_encoder, image_encoder = get_location_and_image_encoders(location_encoder, device=device)

    print(f"Loading dataset from: {data_dir}")
    dataset = S2GeoDataModule(data_dir=data_dir, mode="both", batch_size=256, crop_size=224)
    dataset.setup()
    train_loader = dataset.val_dataloader()
    #norm_factor = get_norm_factor(train_loader)

    global_idx = 0

    header = build_csv_header(autoencoder.dict_size)
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        visual_embeddings = []
        for batch in tqdm(train_loader, desc="Encoding points", unit="batch"):
            points = batch["point"].to(device=device).double()
            print(batch.keys())
            filenames = batch["filename"] 
            image = batch["image"].to(device=device).float()
            with t.no_grad():
                location_embedding = location_encoder(points)
                sparse_activations = autoencoder.encode(location_embedding, use_threshold=True).cpu()

                visual_features = image_encoder(image).cpu() if image is not None else None 
                visual_embeddings.append(visual_features.detach().cpu())

            points_cpu = points.cpu().numpy()
            
            sparse_numpy = sparse_activations.numpy()

            for i, ((lon, lat), act_row) in enumerate(zip(points_cpu, sparse_numpy)):
                writer.writerow([filenames[i], lon, lat,  *act_row.tolist()])
            
        visual_embeddings = t.cat(visual_embeddings, dim=0)
        t.save(visual_embeddings, visual_embeddings_path)


    print(f"Saved sparse activations to: {output_csv}")
    print(f"Saved visual embeddings to: {visual_embeddings_path}")


if __name__ == "__main__":
    args = parse_args()
    export_activations(data_dir=args.data_dir, sae_path=args.sae_path, location_encoder=args.location_encoder)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import os

import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from geoclip.model.image_encoder import ImageEncoder


from saes import BatchTopKSAE

from location_encoders.location_encoder import _load_model, get_visual_encoder, LOCATION_EMBEDDING_DIMENSIONS
from location_encoders.satclip.datamodules.s2geo_dataset import S2GeoDataModule
from eval_data.eval_datataset import GeoClipImageryDataset
from eval_data.eval_datataset import geo_clip_img_val_transform

from training import get_norm_factor
from monosemanticity import monosemanticity

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
        description="Export sparse SAE activations for a dataset."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="The name of the dataset.",
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root folder of the dataset containing index.csv and satellite imagery.",
    )
    parser.add_argument(
        "--location_encoder",
        default="satclip",
        help="The class of the location encoder",
    )
    parser.add_argument(
        "--sae-path",
        required=True,
        help="Path to the pretrained SAE checkpoint file (.pt).",
    )
    return parser.parse_args()

def build_csv_header(dict_size):
    return ["filename","lon", "lat"] + [f"act{i+1}" for i in range(dict_size)]


def get_data_loader(dataset, data_dir):
    print(f"Loading {dataset} dataset from: {data_dir}")
    if dataset == "s2-100k":
        dataset = S2GeoDataModule(data_dir=data_dir, mode="both", batch_size=1024, crop_size=224)
        dataset.setup()
        return dataset.val_dataloader()
    elif dataset == "geoyfcc":
        dataset = GeoClipImageryDataset(dataset_file=os.path.join(data_dir, "sampled_index.csv"), dataset_folder=data_dir, transform=geo_clip_img_val_transform())
        return DataLoader(dataset, batch_size=300, shuffle=False, num_workers=6)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

def export_activations(dataset, data_dir, sae_path, location_encoder):
    sae_model_path = os.path.join(sae_path, "best_ae.pt")
    sparse_activations_output_path = os.path.join(sae_path, "xAI", "sparse_activations.csv")
    print(f"Output CSV will be saved to: {sparse_activations_output_path}")
    os.makedirs(os.path.dirname(sparse_activations_output_path), exist_ok=True)

    visual_embeddings_path = os.path.join(sae_path, "xAI", "visual_embeddings.pt")

    print(f"Using device: {device}")
    print(f"Loading pretrained SAE from: {sae_model_path}")
    autoencoder = BatchTopKSAE.from_pretrained(sae_model_path, device=device).double()
    autoencoder.eval()

    print("Initializing geo and image encoder...")
    location_encoder = _load_model(location_encoder, device=device)
    image_encoder = get_visual_encoder(dataset, device)

    data_loader = get_data_loader(dataset, data_dir)

    global_idx = 0

    header = build_csv_header(autoencoder.dict_size)
    with open(sparse_activations_output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        visual_embeddings = []
        for batch in tqdm(data_loader, desc="Encoding points", unit="batch"):
            points = batch["point"].to(device=device).double()
            filenames = batch["filename"] 
            image = batch["image"].to(device=device).float()
            with t.no_grad():
                location_embedding = location_encoder(points)
                sparse_activations = autoencoder.encode(location_embedding, use_threshold=True).cpu()

                if isinstance(image_encoder, ImageEncoder):
                    print("Extracting visual features using Geoclip image encoder...")
                    out = image_encoder.CLIP.get_image_features(pixel_values=image)
                    feats = out.pooler_output if hasattr(out, "pooler_output") else out
                    visual_features = image_encoder.mlp(feats).cpu()
                else:
                    visual_features = image_encoder(image).cpu() if image is not None else None 
                visual_embeddings.append(visual_features.detach().cpu())

            points_cpu = points.cpu().numpy()
            
            sparse_numpy = sparse_activations.numpy()

            for i, ((lon, lat), act_row) in enumerate(zip(points_cpu, sparse_numpy)):
                writer.writerow([filenames[i], lon, lat,  *act_row.tolist()])
            
        visual_embeddings = t.cat(visual_embeddings, dim=0)
        t.save(visual_embeddings, visual_embeddings_path)


    print(f"Saved sparse activations to: {sparse_activations_output_path}")
    print(f"Saved visual embeddings to: {visual_embeddings_path}")
    return sparse_activations_output_path, visual_embeddings_path


if __name__ == "__main__":
    args = parse_args()
    sparse_activations_path, visual_embeddings_path = export_activations(
        dataset = args.dataset,
        data_dir=args.data_dir,
        sae_path=args.sae_path,
        location_encoder=args.location_encoder)
    monosemanticity(visual_embeddings_path, sparse_activations_path)

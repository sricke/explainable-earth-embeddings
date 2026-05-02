#!/usr/bin/env python3
"""Precompute text and location embeddings for SkyScript splits.

Saves to ~/data/skyscript/{location_model}/{text_model}/{split}.parquet
with columns: lat, lon, text, location_embedding, text_embedding

Run from project root:
    python data/skyscript/precompute_embeddings.py [--location_model satclip] [--text_model open_clip]
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import GeoTextDataset
from models.location_encoder import LocationEncoder
from models.text_encoder import TextEncoder

SKYSCRIPT_DIR = Path.home() / "data" / "skyscript"
SPLITS = ["train", "val", "test"]
DEVICE = "cuda"


def precompute(location_model: str, text_model: str, batch_size: int):
    loc_encoder = LocationEncoder(location_model=location_model).to(DEVICE).eval()
    text_encoder = TextEncoder(text_model=text_model, finetune_mode="only_proj").to(DEVICE).eval()

    out_dir = SKYSCRIPT_DIR / location_model / text_model
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dataset = GeoTextDataset(root=SKYSCRIPT_DIR, split=split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        loc_embs, text_embs = [], []
        with torch.no_grad():
            for latlons, texts in tqdm(loader, desc=split):
                loc_embs.append(loc_encoder(latlons.to(DEVICE)).cpu().float().numpy())
                text_embs.append(text_encoder.encode_texts(list(texts)).cpu().float().numpy())

        loc_embs = np.concatenate(loc_embs)
        text_embs = np.concatenate(text_embs)

        dataset.df[["lat", "lon", "text"]].to_parquet(out_dir / f"{split}.parquet", index=False)
        np.save(out_dir / f"{split}_location_embeddings.npy", loc_embs)
        np.save(out_dir / f"{split}_text_embeddings.npy", text_embs)
        print(f"  -> {out_dir / f'{split}.parquet'}  loc_dim={loc_embs.shape[1]}  text_dim={text_embs.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location_model", default="satclip", choices=["satclip", "geoclip"])
    parser.add_argument("--text_model", default="open_clip", choices=["open_clip", "geoclip"])
    parser.add_argument("--batch_size", type=int, default=2048)
    args = parser.parse_args()

    precompute(args.location_model, args.text_model, args.batch_size)

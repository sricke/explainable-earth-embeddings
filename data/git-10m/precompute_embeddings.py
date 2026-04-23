#!/usr/bin/env python3
"""Precompute text and location embeddings for Git-10M splits.

Saves to ~/data/git-10M/{location_model}/{text_model}/{split}.parquet
with columns: lat, lon, text, location_embedding, text_embedding

Run from project root:
    python data/git-10m/precompute_embeddings.py [--location_model satclip] [--text_model open_clip]
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset import GeoTextDataset
from models.location_encoder import LocationEncoder
from models.text_encoder import TextEncoder

GIT10M_DIR = Path("../../../../data/expanded-git-10M")
SPLITS = ["train", "val", "test"]
DEVICE = "cuda"


def make_collate_fn(tokenizer):
    # Pad every batch to exactly 77 tokens. Variable-length padding produces
    # different tensor shapes each batch, which fragments the CUDA allocator
    # and causes OOM after O(70) batches even with headroom to spare.
    # CLIP's causal mask ensures EOS doesn't attend to trailing PAD tokens,
    # so outputs are numerically identical to variable-length padding.
    def collate_fn(batch):
        latlons, texts = zip(*batch)
        tokens = tokenizer(
            list(texts),
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        return torch.stack(latlons), dict(tokens)
    return collate_fn


def precompute(location_model: str, text_model: str, batch_size: int, cache_every: int = 100):
    loc_encoder = LocationEncoder(location_model=location_model, finetune_mode="only_proj", precomputed=False).to(DEVICE).eval()
    text_encoder = TextEncoder(text_model=text_model, finetune_mode="only_proj", precomputed=False).to(DEVICE).eval()

    out_dir = GIT10M_DIR / location_model / text_model
    out_dir.mkdir(parents=True, exist_ok=True)

    collate_fn = make_collate_fn(text_encoder.tokenizer)

    for split in SPLITS:
        dataset = GeoTextDataset(root=GIT10M_DIR, split=split)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            collate_fn=collate_fn,
        )

        cache_dir = out_dir / f"{split}_cache"
        cache_dir.mkdir(exist_ok=True)

        loc_buf, text_buf, chunk_idx = [], [], 0

        def flush():
            nonlocal loc_buf, text_buf, chunk_idx
            np.save(cache_dir / f"loc_{chunk_idx:05d}.npy", np.concatenate(loc_buf))
            np.save(cache_dir / f"text_{chunk_idx:05d}.npy", np.concatenate(text_buf))
            loc_buf, text_buf, chunk_idx = [], [], chunk_idx + 1

        with torch.inference_mode():
            for i, (latlons, tokens) in enumerate(tqdm(loader, desc=split)):
                loc_buf.append(loc_encoder(latlons.to(DEVICE)).cpu().float().numpy())
                # bfloat16 autocast halves CLIP activation memory (~5 GB → ~2.5 GB
                # at batch 8192) and uses A100 Tensor Cores for ~2x throughput.
                text_buf.append(text_encoder.encode_texts(tokens).cpu().float().numpy())
                if (i + 1) % cache_every == 0:
                    flush()
                    torch.cuda.empty_cache()

        if loc_buf:
            flush()

        loc_embs = np.concatenate([np.load(f) for f in sorted(cache_dir.glob("loc_*.npy"))])
        text_embs = np.concatenate([np.load(f) for f in sorted(cache_dir.glob("text_*.npy"))])

        out_df = dataset.df[["lat", "lon", "text"]].copy()
        out_df["location_embedding"] = list(loc_embs)
        out_df["text_embedding"] = list(text_embs)
        out_df.to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"  -> {out_dir / f'{split}.parquet'}  loc_dim={loc_embs.shape[1]}  text_dim={text_embs.shape[1]}")

        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location_model", default="satclip", choices=["satclip", "geoclip"])
    parser.add_argument("--text_model", default="open_clip_vit_l", choices=["open_clip_vit_l", "open_clip_vit_h", "geoclip"])
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--cache_every", type=int, default=100, help="Flush embeddings to disk every N batches")
    args = parser.parse_args()

    precompute(args.location_model, args.text_model, args.batch_size, args.cache_every)

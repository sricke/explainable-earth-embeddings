#!/usr/bin/env python3
"""Precompute text and location embeddings for Git-10M splits.

Saves to ~/data/git-10M/{location_model}/{text_model}/{split}_location_embeddings.npy
and {split}_text_embeddings.npy

Run from project root:
    python data/git-10m/precompute_embeddings.py [--location_model satclip] [--text_model open_clip]
"""

import os
import sys
import shutil
import argparse
from itertools import islice
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
    out_dir = GIT10M_DIR / location_model / text_model
    out_dir.mkdir(parents=True, exist_ok=True)

    loc_encoder = text_encoder = collate_fn = None  # lazy: only load if GPU compute needed

    for split in SPLITS:
        print(f"\n[{split}]")
        if (out_dir / f"{split}_location_embeddings.npy").exists() and \
           (out_dir / f"{split}_text_embeddings.npy").exists():
            print(f"  Skipping (npy files exist)")
            continue

        cache_dir = out_dir / f"{split}_cache"
        cache_dir.mkdir(exist_ok=True)
        dataset = GeoTextDataset(root=GIT10M_DIR, split=split)
        print(f"  Dataset: {len(dataset):,} samples")

        existing_loc = sorted(cache_dir.glob("loc_*.npy"))
        chunk_idx = len(existing_loc)
        rows_cached = sum(np.load(f, mmap_mode="r").shape[0] for f in existing_loc)
        print(f"  Cache: {chunk_idx} chunks, {rows_cached:,} rows already computed")

        if rows_cached < len(dataset):
            if loc_encoder is None:
                print("  Loading models onto GPU...")
                loc_encoder = LocationEncoder(location_model=location_model, finetune_mode="only_proj", precomputed=False).to(DEVICE).eval()
                text_encoder = TextEncoder(text_model=text_model, finetune_mode="only_proj", precomputed=False).to(DEVICE).eval()
                collate_fn = make_collate_fn(text_encoder.tokenizer)
                print("  Models loaded.")

            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                pin_memory=True, prefetch_factor=4, collate_fn=collate_fn)
            skip = chunk_idx * cache_every
            loc_buf, text_buf = [], []

            def flush():
                nonlocal chunk_idx
                np.save(cache_dir / f"loc_{chunk_idx:05d}.npy", np.concatenate(loc_buf))
                np.save(cache_dir / f"text_{chunk_idx:05d}.npy", np.concatenate(text_buf))
                loc_buf.clear(); text_buf.clear()
                chunk_idx += 1

            with torch.inference_mode():
                pbar = tqdm(islice(loader, skip, None), initial=skip, total=len(loader),
                            desc=f"  {split}", unit="batch", dynamic_ncols=True)
                for i, (latlons, tokens) in enumerate(pbar, skip):
                    loc_buf.append(loc_encoder(latlons.to(DEVICE)).cpu().float().numpy())
                    # bfloat16 autocast halves CLIP activation memory (~5 GB → ~2.5 GB
                    # at batch 8192) and uses A100 Tensor Cores for ~2x throughput.
                    text_buf.append(text_encoder.encode_texts(tokens).cpu().float().numpy())
                    if (i + 1) % cache_every == 0:
                        flush()
                        torch.cuda.empty_cache()
                pbar.close()

            if loc_buf:
                flush()

        # Consolidate cache chunks into final memmaps
        from numpy.lib.format import open_memmap
        all_chunks = sorted(cache_dir.glob("loc_*.npy"))
        first_loc  = np.load(all_chunks[0], mmap_mode='r')
        first_text = np.load(all_chunks[0].parent / all_chunks[0].name.replace("loc_", "text_"), mmap_mode='r')
        loc_out  = open_memmap(out_dir / f"{split}_location_embeddings.npy", mode='w+', dtype='float32', shape=(len(dataset), first_loc.shape[1]))
        text_out = open_memmap(out_dir / f"{split}_text_embeddings.npy",     mode='w+', dtype='float32', shape=(len(dataset), first_text.shape[1]))
        row_cursor = 0
        for loc_f in tqdm(all_chunks, desc=f"  {split} writing", unit="chunk", dynamic_ncols=True):
            loc_e  = np.load(loc_f)
            text_e = np.load(loc_f.parent / loc_f.name.replace("loc_", "text_"))
            n = len(loc_e)
            loc_out[row_cursor:row_cursor + n]  = loc_e
            text_out[row_cursor:row_cursor + n] = text_e
            row_cursor += n
        del loc_out, text_out  # flush

        shutil.rmtree(cache_dir)
        print(f"  -> {out_dir / split}_{{location,text}}_embeddings.npy  loc_dim={first_loc.shape[1]}  text_dim={first_text.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--location_model", default="satclip", choices=["satclip", "geoclip"])
    parser.add_argument("--text_model", default="open_clip_vit_l", choices=["open_clip_vit_l", "open_clip_vit_h", "geoclip"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--cache_every", type=int, default=100, help="Flush embeddings to disk every N batches")
    args = parser.parse_args()

    precompute(args.location_model, args.text_model, args.batch_size, args.cache_every)

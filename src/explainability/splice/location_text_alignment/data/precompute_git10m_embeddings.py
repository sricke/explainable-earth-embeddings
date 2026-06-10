import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dataset import GeoTextDataset
from models.location_encoder import LocationEncoder
from torch.utils.data import DataLoader

SPLITS = ["train", "val", "test"]
DEVICE = "cuda"


def precompute_location_embeddings(
    location_model: str, git10m_dir: Path, out_dir: Path, batch_size: int
):
    # Frozen encoder (no projection head)
    loc_encoder = (
        LocationEncoder(
            location_model=location_model,
            finetune_mode="only_proj",
            precomputed=False,
        )
        .to(DEVICE)
        .eval()
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        dataset = GeoTextDataset(
            root=git10m_dir, split=split, precomputed_location_embeddings=False
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        loc_embs = []
        with torch.no_grad(), torch.autocast("cuda"):
            for latlons, _ in tqdm(loader, desc=split):
                loc_embs.append(loc_encoder(latlons.to(DEVICE)).cpu().float().numpy())

        loc_embs = np.concatenate(loc_embs)

        out_df = dataset.df[["lat", "lon", "text"]].copy()
        out_df["location_embedding"] = list(loc_embs)
        out_df.to_parquet(out_dir / f"{split}.parquet", index=False)
        print(f"  -> {out_dir / f'{split}.parquet'}  loc_dim={loc_embs.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--git10m_dir", required=True)
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory; defaults to <git10m_dir>/<location_model>/",
    )
    parser.add_argument(
        "--location_model",
        required=True,
        choices=["satclip", "geoclip", "climplicit", "csp_fmow", "sinr"],
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    git10m_dir = Path(args.git10m_dir)
    out_dir = Path(args.out_dir) if args.out_dir else git10m_dir / args.location_model

    precompute_location_embeddings(
        args.location_model, git10m_dir, out_dir, args.batch_size
    )

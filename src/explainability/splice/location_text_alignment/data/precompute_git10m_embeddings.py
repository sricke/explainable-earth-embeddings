import argparse
import gc
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
        )

        out_path = out_dir / f"{split}.parquet"
        tmp_path = out_path.with_suffix(".parquet.tmp")
        meta = dataset.df[["lat", "lon", "text"]]

        # Stream each batch to disk so host memory stays flat; shuffle=False
        # keeps batches aligned with the dataframe rows.
        writer = None
        start = 0
        loc_dim = None
        with torch.no_grad(), torch.autocast("cuda"):
            for latlons, _ in tqdm(loader, desc=split):
                emb = loc_encoder(latlons.to(DEVICE)).cpu().float().numpy()
                loc_dim = emb.shape[1]
                n = len(emb)
                batch_df = meta.iloc[start : start + n].reset_index(drop=True)
                start += n

                table = pa.Table.from_pandas(batch_df, preserve_index=False)
                table = table.append_column(
                    "location_embedding",
                    pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), loc_dim),
                )
                if writer is None:
                    writer = pq.ParquetWriter(tmp_path, table.schema)
                writer.write_table(table)
        writer.close()
        tmp_path.rename(out_path)
        print(f"  -> {out_path}  loc_dim={loc_dim}")

        del loader, dataset, meta
        gc.collect()
        torch.cuda.empty_cache()


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

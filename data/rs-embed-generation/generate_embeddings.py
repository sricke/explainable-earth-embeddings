#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from rs_embed import get_embeddings_batch, PointBuffer, OutputSpec, TemporalSpec

import ee
ee.Authenticate()
ee.Initialize(project="avid-poet-486323-a3")

DEFAULT_IMG_SIDE = 256
DEFAULT_BATCH_SIZE = 64
TEMPORAL = TemporalSpec(mode="year", year=2022) # maybe change??


def parse_resolution(res_str: str) -> float:
    # Resolution is in a form like {N}m/pix
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*m/pix", res_str.strip())
    assert m, f"Expected resolution like '10.0m/pix', got {res_str!r}"
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input CSV or Parquet with lat, lon, resolution columns")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--emb_model", required=True, help="rs_embed model identifier")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    path = Path(args.file)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    assert "lat" in df.columns and "lon" in df.columns and "resolution" in df.columns

    spatials = [
        PointBuffer(
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            buffer_m=parse_resolution(str(row["resolution"])) * DEFAULT_IMG_SIDE / 2, # double check this is right ... what exactly is buffer_m?
        )
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating spatial points")
    ]

    all_embeddings = []
    for i in tqdm(range(0, len(spatials), args.batch_size), desc="Generating embeddings"):
        batch = spatials[i : i + args.batch_size]
        results = get_embeddings_batch(
            args.emb_model,
            spatials=batch,
            temporal=TEMPORAL, # might need to change depending on model ...
            output=OutputSpec.pooled(), # I think pooled makes sense here if using the resolution as buffer_m, but double check
        )
        for emb in results:
            vec = emb.data if hasattr(emb, "data") else np.asarray(emb)
            all_embeddings.append(np.asarray(vec, dtype=np.float32))

    df["location_embedding"] = all_embeddings

    if save_path.suffix == ".parquet":
        df.to_parquet(save_path, index=False)
    else:
        df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()

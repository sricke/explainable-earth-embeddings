import argparse
import pandas as pd
from pathlib import Path

SEED = 42
N = 100_000

# yfcc_metadata is tab-separated (YFCC100M format); longitude at index 11, latitude at index 12
LON_IDX = 11
LAT_IDX = 12

parser = argparse.ArgumentParser()
parser.add_argument("--geoyfcc_path", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

df = pd.read_pickle(args.geoyfcc_path)

coords = df["yfcc_metadata"].str.split("\t", expand=True)
lons = pd.to_numeric(coords[LON_IDX], errors="coerce")
lats = pd.to_numeric(coords[LAT_IDX], errors="coerce")

result = pd.DataFrame({"lat": lats, "lon": lons}).dropna()

if len(result) > N:
    result = result.sample(n=N, random_state=SEED)

result = result.reset_index(drop=True)

args.out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.out, index=False)
print(f"Saved {len(result)} points -> {args.out}")

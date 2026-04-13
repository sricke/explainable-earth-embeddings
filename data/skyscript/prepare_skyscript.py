import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SKYSCRIPT_DIR = Path("/home/libe2152/data/skyscript")
META_DIR = SKYSCRIPT_DIR / "meta2"
OUT_DIR = SKYSCRIPT_DIR

SPLITS = {
    "train": SKYSCRIPT_DIR / "SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv",
    "val":   SKYSCRIPT_DIR / "SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv",
    "test":  SKYSCRIPT_DIR / "SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv",
}


def filepath_to_pickle(filepath: str) -> Path:
    # example: images6/a260343570_CH_18.jpg -> meta2/a260343570_CH_18.pickle
    stem = Path(filepath).stem
    return META_DIR / f"{stem}.pickle"


def bbox_to_centroid(bbox: tuple) -> tuple[float, float]:
    # (west, south, east, north) as per skyscript github
    west, south, east, north = bbox
    lat = (south + north) / 2
    lon = (west + east) / 2
    return lat, lon


def process_split(name: str, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    rows = []
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        pkl_path = filepath_to_pickle(row["filepath"])
        if not pkl_path.exists():
            missing += 1
            continue
        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)
        lat, lon = bbox_to_centroid(meta["bbox"])
        rows.append({"lat": lat, "lon": lon, "text": row["title_multi_objects"]})

    out_df = pd.DataFrame(rows, columns=["lat", "lon", "text"])
    out_path = OUT_DIR / f"{name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"{name}: {len(out_df)} rows saved to {out_path} ({missing} missing pickles skipped)")


if __name__ == "__main__":
    for name, csv_path in SPLITS.items():
        process_split(name, csv_path)

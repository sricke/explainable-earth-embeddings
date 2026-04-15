from datasets import load_from_disk
import os

path = "~/data/git-10M"
out_dir = os.path.expanduser("~/data/git-10M-no-images")

ds = load_from_disk(path)

os.makedirs(out_dir, exist_ok=True)

for split_name, split in ds.items():
    # remove image column
    split = split.remove_columns(["image"])

    out_path = os.path.join(out_dir, f"{split_name}.parquet")

    # write parquet
    split.to_parquet(out_path)

    print(f"Saved {split_name} -> {out_path}")
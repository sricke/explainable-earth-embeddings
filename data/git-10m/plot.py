import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

SPLITS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

def make_basemap(ax):
    m = Basemap(projection='cyl', resolution='c', ax=ax)
    m.drawcoastlines()
    return m

# Load all splits once
data = {
    split: pd.read_parquet(f"/home/libe2152/data/git-10M/{split}.parquet", columns=["lon", "lat"])
    for split in SPLITS
}

# One map per split
for split, color in SPLITS.items():
    df = data[split]
    fig, ax = plt.subplots(1, figsize=(10, 5))
    make_basemap(ax)
    ax.scatter(df.lon, df.lat, s=1, alpha=0.1, color=color, label=f"{split} (n={len(df):,})")
    ax.legend(markerscale=5)
    ax.set_title(split)
    plt.tight_layout()
    plt.savefig(f"git10m_{split}_map.png", dpi=150, bbox_inches='tight')
    plt.close()

# Combined map
fig, ax = plt.subplots(1, figsize=(10, 5))
make_basemap(ax)
for split, color in SPLITS.items():
    df = data[split]
    ax.scatter(df.lon, df.lat, s=1, alpha=0.1, color=color, label=f"{split} (n={len(df):,})")
    
ax.legend(markerscale=5)
ax.set_title("all splits")
plt.tight_layout()
plt.savefig("git10m_map.png", dpi=150, bbox_inches='tight')
plt.close()

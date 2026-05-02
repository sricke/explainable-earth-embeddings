import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

SPLITS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

fig, ax = plt.subplots(1, figsize=(10, 5))
m = Basemap(projection='cyl', resolution='c', ax=ax)
m.drawcoastlines()

for split, color in SPLITS.items():
    df = pd.read_csv(f"/home/libe2152/data/skyscript/{split}.csv")
    ax.scatter(df.lon, df.lat, s=2, alpha=1, color=color, label=split)

ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig("skyscript_map.png", dpi=150, bbox_inches='tight')

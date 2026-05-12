import sys
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from paths import DATA_ROOT

df = pd.read_csv(DATA_ROOT / "dense_grid" / "dense_grid.csv")

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(14, 7))
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.COASTLINE, linewidth=0.4)

ax.scatter(
    df["lon"], df["lat"],
    s=0.3, c="steelblue", alpha=0.4,
    transform=ccrs.PlateCarree(),
)

ax.set_global()
ax.set_title(f"Dense grid ({len(df):,} points)")
plt.tight_layout()
plt.savefig(DATA_ROOT / "dense_grid" / "dense_grid.png", dpi=150)
plt.show()

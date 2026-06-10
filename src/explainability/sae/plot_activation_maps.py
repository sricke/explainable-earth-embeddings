#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot world maps for each non-sparse activation column in a CSV file."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV file with columns lon, lat, act1, act2, ..., actN.",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=10,
        help="Minimum number of positive activations required to plot a feature.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional limit on the number of activation columns to plot.",
    )
    return parser.parse_args()


def build_axes():
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Activation map")
        ax.grid(True, linestyle="--", alpha=0.4)
        return fig, ax


def save_activation_maps(input_csv, min_positive, max_features):
    print("AAAA")
    marker_size = 8.0
    output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    if "lon" not in df.columns or "lat" not in df.columns:
        raise ValueError("Input CSV must contain 'lon' and 'lat' columns.")

    #activation_columns = [c for c in df.columns if c not in {"lon", "lat", "index", "filename"}]
    activation_columns = ["act406"]
    print(activation_columns)
    if not activation_columns:
        raise ValueError("No activation columns found in the CSV.")

    print(f"Found {len(activation_columns)} activation columns.")
    plotted = 0

    for i, col in enumerate(activation_columns):
        if max_features is not None and plotted >= max_features:
            break

        values = df[col].to_numpy(dtype=float)
        positive_mask = values > 0
        n_positive = int(np.sum(positive_mask))
        if n_positive < min_positive:
            print(f"Skipping {col}: only {n_positive} positive values.")
            continue

        lon = df.loc[positive_mask, "lon"].to_numpy(dtype=float)
        lat = df.loc[positive_mask, "lat"].to_numpy(dtype=float)
        plot_values = values[positive_mask]

        # Scale marker size by activation strength so stronger activations appear thicker.
        if plot_values.size > 1:
            normalized_strength = (plot_values - plot_values.min()) / (plot_values.max() - plot_values.min())
        else:
            normalized_strength = np.ones_like(plot_values)
        marker_sizes = marker_size * (1.0 + 6.0 * normalized_strength)

        fig, ax = build_axes()
        scatter_kwargs = dict(
            c=plot_values,
            cmap="plasma",
            s=marker_sizes,
            alpha=0.7,
            edgecolors="none",
        )
        if HAS_CARTOPY:
            scatter_kwargs["transform"] = ccrs.PlateCarree()

        sc = ax.scatter(lon, lat, **scatter_kwargs)

        ax.set_title(f"{col} activations (>0) on world map")
        cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.036, pad=0.04)
        cbar.set_label(col)

        outfile = os.path.join(output_dir, f"{col}.png")
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {outfile} ({n_positive} points)")
        plotted += 1

    if plotted == 0:
        print("No activation columns were plotted. Check min_positive or the CSV values.")
    else:
        print(f"Plotted {plotted} activation maps.")


if __name__ == "__main__":
    args = parse_args()
    save_activation_maps(
        input_csv=args.input_csv,
        min_positive=args.min_positive,
        max_features=args.max_features,
    )

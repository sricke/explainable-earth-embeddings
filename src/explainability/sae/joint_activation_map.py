#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Attempt to import Cartopy for geographic plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a wide map where point size represents activation strength."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the CSV file containing lon, lat, and activation columns.",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=5,
        help="Minimum positive values required for a neuron to be plotted.",
    )
    return parser.parse_args()

def save_wide_scaled_map(input_csv, min_positive):
    output_dir = os.path.dirname(input_csv)
    df = pd.read_csv(input_csv)
    
    activation_columns = ["act872", "act395", "act294"]

    cmap = plt.get_cmap("tab10")
    
    size_factor = 5.0 
    
    # Wide aspect ratio: (Width, Height)
    fig = plt.figure(figsize=(10, 4)) 
    
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.spines['geo'].set_linewidth(0.5) 
        ax.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
        ax.coastlines(linewidth=0.4, zorder=1)
        ax.set_global() 
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_axis_off() 

    plotted_count = 0
    for i, col in enumerate(activation_columns):
        if col not in df.columns:
            continue
            
        mask = df[col] > 0
        if mask.sum() < min_positive:
            continue

        numeric_match = re.search(r'\d+', col)
        if numeric_match:
            neuron_num = int(numeric_match.group()) - 1
            new_label = f'Neuron {neuron_num}'
        else:
            new_label = col

        active_data = df.loc[mask]
        
        scaled_sizes = active_data[col] * size_factor

        scatter_kwargs = {
            "s": scaled_sizes, 
            "color": cmap(i % 10),
            "alpha": 0.6, # Slightly lower alpha helps see overlapping scaled points
            "label": new_label,
            "edgecolors": "none",
            "zorder": 2 + i
        }
        
        if HAS_CARTOPY:
            scatter_kwargs["transform"] = ccrs.PlateCarree()

        ax.scatter(active_data["lon"], active_data["lat"], **scatter_kwargs)
        plotted_count += 1

    if plotted_count > 0:
        
        leg = ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, -0.05),
            ncol=plotted_count, 
            prop={'weight': 'bold', 'size': 8},
            frameon=False,
            handletextpad=0.4,
            columnspacing=2.5
        )
        
        for handle in leg.legend_handles:
            handle.set_sizes([30.0])

        outfile = os.path.join(output_dir, "neuron_map_scaled.png")
        fig.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Successfully saved wide scaled map to: {outfile}")
    else:
        print("No activation columns met the plotting criteria.")

if __name__ == "__main__":
    args = parse_args()
    save_wide_scaled_map(args.input_csv, args.min_positive)
import argparse
import os
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

def plot_neuron_activations_on_map(input_csv_s2100k, input_csv_geoyfcc, activation_columns_s2100k, activation_columns_geoyfcc, min_positive):
    output_dir = os.path.dirname(input_csv_s2100k) or os.getcwd()
    df_s2100k = pd.read_csv(input_csv_s2100k)
    df_geoyfcc = pd.read_csv(input_csv_geoyfcc)
    
    act_s2 = [c.strip() for c in activation_columns_s2100k.split(",") if c.strip()]
    act_geo = [c.strip() for c in activation_columns_geoyfcc.split(",") if c.strip()]
    
    dataset_configs = [
        ("S2-100k", df_s2100k, "o", act_s2, 4.0), 
        ("Geo-YFCC", df_geoyfcc, "D", act_geo, 8.0)
    ]

    unique_neurons = []
    neuron_to_marker = {}
    for _, _, marker, cols, _ in dataset_configs:
        for col in cols:
            if col not in unique_neurons:
                unique_neurons.append(col)
                neuron_to_marker[col] = marker

    custom_colors = {'act758': '#ffaa00'}
    hsv_cmap = plt.get_cmap("hsv")
    gen_colors = hsv_cmap(np.linspace(0, 1, len(unique_neurons), endpoint=False))
    neuron_to_color = {n: custom_colors.get(n, gen_colors[i]) for i, n in enumerate(unique_neurons)}

    fig = plt.figure(figsize=(8, 11))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) if HAS_CARTOPY else fig.add_subplot(1, 1, 1)
    
    if HAS_CARTOPY:
        ax.add_feature(cfeature.LAND, facecolor="#eeeeee"); ax.coastlines(linewidth=0.4); ax.set_global()
    else:
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_axis_off()

    for label, df, ds_marker, cols, scale_factor in dataset_configs:
        for col in cols:
            if col not in df.columns: continue
            mask = df[col] > 0
            if mask.sum() < min_positive: continue
            
            kwargs = {"s": df.loc[mask, col] * scale_factor, "color": neuron_to_color[col], 
                      "alpha": 0.8, "marker": ds_marker, "edgecolors": 'black', "linewidth": 0.5}
            if HAS_CARTOPY: kwargs["transform"] = ccrs.PlateCarree()
            ax.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"], **kwargs)

    def _format_label(c): 
        m = re.search(r"\d+", c)
        return f"Neuron {int(m.group()) - 1}" if m else c

    # Neuron Legend - Font size set to 8.5
    neuron_handles = [Line2D([0], [0], marker=neuron_to_marker[n], color='w', 
                             markerfacecolor=neuron_to_color[n], markeredgecolor='black', 
                             markeredgewidth=0.5, markersize=6) for n in unique_neurons]
    
    leg1 = ax.legend(neuron_handles, [_format_label(n) for n in unique_neurons],
                     loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=6,
                     prop={'size': 8.5}, frameon=False,
                     columnspacing=0.5, handletextpad=0.2, borderpad=0)
    ax.add_artist(leg1)

    # Dataset Legend
    ds_handles = [Line2D([0], [0], marker=cfg[2], color='w', markerfacecolor='k', 
                         markeredgecolor='black', markersize=6) for cfg in dataset_configs]
    
    ax.legend(ds_handles, [cfg[0] for cfg in dataset_configs],
              loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=2, 
              prop={'weight': 'normal', 'size': 9}, frameon=False)

    outfile = os.path.join(output_dir, "neuron_map_final_final.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv_s2100k", required=True)
    parser.add_argument("--input_csv_geoyfcc", required=True)
    parser.add_argument("--activation_columns_s2100k", default="act242,act758,act103")
    parser.add_argument("--activation_columns_geoyfcc", default="act732,act31,act824")
    parser.add_argument("--min-positive", type=int, default=5)
    args = parser.parse_args()
    plot_neuron_activations_on_map(args.input_csv_s2100k, args.input_csv_geoyfcc, 
                                   args.activation_columns_s2100k, args.activation_columns_geoyfcc, 
                                   args.min_positive)
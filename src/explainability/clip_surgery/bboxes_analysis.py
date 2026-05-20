import os
import shutil
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torch import nn
from torchgeo.models import resnet50
from torchgeo.models import ResNet50_Weights as Torchgeo_ResNet50_Weights
from torchgeo.models import resnet50 as Torchgeo_resnet50
from torchvision.models import ResNet50_Weights as Torchvision_ResNet50_Weights
from torchvision.models import resnet50 as Torchvision_resnet50
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--place", type=str, default="Eifel")
parser.add_argument("--output_dir", type=str, default=os.path.join(_THIS_DIR, "out"))
parser.add_argument("--bboxes_subdir", type=str, default="bboxes")
parser.add_argument("--model", type=str, default="resnet50", choices=["satclip", "resnet50"])
parser.add_argument("--weights", type=str, default="imagenet", choices=["imagenet", "moco"])
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--perplexity", type=int, default=10)
parser.add_argument("--eps", type=float, default=4)
args = parser.parse_args()

def _draw_images(ax, xlo, xhi, ylo, yhi, zoom, label_fontsize=7):
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    in_cell = (
        (X_2d[:, 0] >= xlo) & (X_2d[:, 0] <= xhi)
        & (X_2d[:, 1] >= ylo) & (X_2d[:, 1] <= yhi)
    )
    thumb_px = 32 * zoom
    y_offset_pts = thumb_px / 2 + 2
    for idx in np.where(in_cell)[0]:
        x, y = X_2d[idx]
        label = labels[idx]
        edge = plt.cm.tab10(label % 10) if label >= 0 else 'lightgray'
        im = OffsetImage(imgs_cache[idx], zoom=zoom)
        ab = AnnotationBbox(
            im, (x, y),
            frameon=True,
            pad=0.1,
            bboxprops=dict(edgecolor=edge, linewidth=1.5),
        )
        ax.add_artist(ab)
        cluster_text = 'noise' if label == -1 else str(label)
        ax.annotate(
            cluster_text,
            xy=(x, y),
            xytext=(0, -y_offset_pts),
            textcoords='offset points',
            ha='center', va='top',
            fontsize=label_fontsize,
            color=edge if label >= 0 else 'dimgray',
            fontweight='bold',
        )


if args.model == "resnet50":
    if args.weights is None:
        raise ValueError("Weights are required for resnet50")
    elif args.weights == "imagenet":
        weights = Torchvision_ResNet50_Weights.IMAGENET1K_V1
        model = Torchvision_resnet50(weights=weights)
        model.fc = nn.Identity()
        preprocess = weights.transforms()
    elif args.weights == "moco":
        weights = Torchgeo_ResNet50_Weights.SENTINEL2_RGB_MOCO
        model = Torchgeo_resnet50(weights=weights, num_classes=0, global_pool='avg')
        preprocess = weights.transforms
    else:
        raise ValueError(f"Invalid weights: {args.weights}")
model.eval()

input_dir = os.path.join(args.output_dir, "satclip", args.place)
bboxes_subdir = args.bboxes_subdir


filenames = []
embeddings = []
for tile in os.listdir(input_dir):
    tile_bboxes_dir = os.path.join(input_dir, tile, bboxes_subdir)
    for bbox in os.listdir(tile_bboxes_dir):
        bbox_path = os.path.join(tile_bboxes_dir, bbox)
        image = Image.open(bbox_path).convert('RGB')
        image = pil_to_tensor(image).float().unsqueeze(0)
        image = preprocess(image)
        with torch.no_grad():
            embedding = model(image)
            
        filenames.append((tile, bbox_path))
        embeddings.append(embedding)

X = torch.cat(embeddings, dim=0).numpy()


# KNN DISTANCE GRAPH

k = args.k  # same as min_samples
nbrs = NearestNeighbors(n_neighbors=k).fit(X)
distances, _ = nbrs.kneighbors(X)

distances = np.sort(distances[:, k-1])

for perplexity in [3,5,7,8,10,12]:
    output_dir = os.path.join(args.output_dir, 'bboxes_analysis', args.place, args.weights, f"perplexity_{perplexity}")
    tsne = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=perplexity, random_state=42)
    X_2d = tsne.fit_transform(X)
    clustering = DBSCAN(eps=args.eps, min_samples=args.k).fit(X_2d)
    labels = clustering.labels_
    print(f"Perplexity: {perplexity}, Clusters: {len(np.unique(labels))}")


    pad_x = 0.05 * (X_2d[:, 0].max() - X_2d[:, 0].min() + 1e-6)
    pad_y = 0.05 * (X_2d[:, 1].max() - X_2d[:, 1].min() + 1e-6)
    xlim = (X_2d[:, 0].min() - pad_x, X_2d[:, 0].max() + pad_x)
    ylim = (X_2d[:, 1].min() - pad_y, X_2d[:, 1].max() + pad_y)

    imgs_cache = [np.asarray(Image.open(p).convert('RGB')) for _, p in filenames]
    
    plots_dir = os.path.join(output_dir, 'plots')
    clusters_dir = os.path.join(output_dir, 'clusters')
    os.makedirs(plots_dir, exist_ok=True)
    if os.path.isdir(clusters_dir):
        shutil.rmtree(clusters_dir)
    os.makedirs(clusters_dir, exist_ok=True)

    plt.plot(distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {k}th neighbor")
    plt.title("K-Distance Graph")
    plt.savefig(os.path.join(plots_dir, 'k_distance_graph.png'), dpi=300)
    plt.close()
    fig_img, ax_img = plt.subplots(figsize=(12, 10))
    _draw_images(ax_img, *xlim, *ylim, zoom=0.5)
    ax_img.set_xlabel('t-SNE 1')
    ax_img.set_ylabel('t-SNE 2')
    fig_img.tight_layout()
    fig_img.savefig(os.path.join(plots_dir, 'overview.png'), dpi=300)

    n_rows, n_cols = 3, 3
    x_edges = np.linspace(xlim[0], xlim[1], n_cols + 1)
    y_edges = np.linspace(ylim[0], ylim[1], n_rows + 1)
    cell_zoom = 1.5

    fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(30, 26))
    for r in range(n_rows):
        for c in range(n_cols):
            xlo, xhi = x_edges[c], x_edges[c + 1]
            ylo, yhi = y_edges[r], y_edges[r + 1]
            in_cell = (
                (X_2d[:, 0] >= xlo) & (X_2d[:, 0] <= xhi)
                & (X_2d[:, 1] >= ylo) & (X_2d[:, 1] <= yhi)
            )
            if in_cell.any():
                pts = X_2d[in_cell]
                cell_pad_x = 0.1 * (pts[:, 0].max() - pts[:, 0].min() + 1e-6)
                cell_pad_y = 0.1 * (pts[:, 1].max() - pts[:, 1].min() + 1e-6)
                view_xlo = pts[:, 0].min() - cell_pad_x
                view_xhi = pts[:, 0].max() + cell_pad_x
                view_ylo = pts[:, 1].min() - cell_pad_y
                view_yhi = pts[:, 1].max() + cell_pad_y
            else:
                view_xlo, view_xhi, view_ylo, view_yhi = xlo, xhi, ylo, yhi

            ax = axes[n_rows - 1 - r, c]
            _draw_images(ax, view_xlo, view_xhi, view_ylo, view_yhi, zoom=cell_zoom)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'x∈[{xlo:.1f},{xhi:.1f}]  y∈[{ylo:.1f},{yhi:.1f}]', fontsize=8)

            fig_cell, ax_cell = plt.subplots(figsize=(20, 16))
            _draw_images(ax_cell, view_xlo, view_xhi, view_ylo, view_yhi, zoom=cell_zoom)
            ax_cell.set_xlabel('t-SNE 1')
            ax_cell.set_ylabel('t-SNE 2')
            ax_cell.set_title(f'cell (r={r}, c={c})  x∈[{xlo:.1f},{xhi:.1f}]  y∈[{ylo:.1f},{yhi:.1f}]')
            fig_cell.tight_layout()
            fig_cell.savefig(os.path.join(plots_dir, f'cell_r{r}_c{c}.png'), dpi=300)
            plt.close(fig_cell)

            fig_grid.suptitle('t-SNE grid (3×3)', y=0.995)
            fig_grid.tight_layout()
            fig_grid.savefig(os.path.join(plots_dir, 'grid.png'), dpi=300)

            for idx, (tile, src_path) in enumerate(filenames):
                label = labels[idx]
                cluster_name = 'noise' if label == -1 else f'cluster_{label:02d}'
                cluster_folder = os.path.join(clusters_dir, cluster_name)
                os.makedirs(cluster_folder, exist_ok=True)
                dst_name = f'{tile}__{os.path.basename(src_path)}'
                shutil.copy2(src_path, os.path.join(cluster_folder, dst_name))

            fig_sc, ax_sc = plt.subplots(figsize=(10, 8))
            noise_mask = labels == -1
            if noise_mask.any():
                ax_sc.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                            c='lightgray', s=40, label='noise')
            if (~noise_mask).any():
                sc = ax_sc.scatter(X_2d[~noise_mask, 0], X_2d[~noise_mask, 1],
                                c=labels[~noise_mask], cmap='tab10', s=40)
            ax_sc.set_xlim(*xlim)
            ax_sc.set_ylim(*ylim)
            ax_sc.set_xlabel('t-SNE 1')
            ax_sc.set_ylabel('t-SNE 2')
            ax_sc.set_title('DBSCAN clusters')
            if noise_mask.any():
                ax_sc.legend()
            fig_sc.tight_layout()
            fig_sc.savefig(os.path.join(plots_dir, 'scatter.png'), dpi=300)

            plt.show()



"""
Plotting utilities for feature selection analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_importance_heatmap(importance_matrices: dict, embeddings: dict,
                             task_names: list, method: str = "lasso",
                             top_k: int = 64, out_path: Path = None):
    """Heatmap of sparse model importance scores (rows=top dims, cols=tasks).

    Args:
        importance_matrices: {emb_name: {"lasso": ndarray(n_dims, n_tasks), "enet": ...}}
        embeddings: dict of embedding arrays (used for ordering only)
        task_names: ordered list of task/dataset names
        method: "lasso" or "enet"
        top_k: number of top dims to show (by mean importance)
        out_path: save path; if None, just shows
    """
    fig, axes = plt.subplots(1, len(embeddings), figsize=(8 * len(embeddings), 14))
    if len(embeddings) == 1:
        axes = [axes]

    for ax, emb_name in zip(axes, embeddings):
        mat = importance_matrices[emb_name][method]   # (n_dims, n_tasks)
        top_idx = np.argsort(mat.mean(axis=1))[::-1][:top_k]
        df_plot = pd.DataFrame(mat[top_idx], index=[f"dim {i}" for i in top_idx], columns=task_names)
        sns.heatmap(df_plot, ax=ax, cmap="magma", linewidths=0, cbar_kws={"label": f"|{method.upper()} coef|"})
        ax.set_title(f"{emb_name} — {method.upper()} Importance\n(top {top_k} dims)", fontsize=12)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_sparsity_barchart(importance_matrices: dict, task_names: list,
                            out_path: Path = None):
    """Bar chart of number of non-zero dims per (embedding × dataset) for LASSO and ENET."""
    records = []
    for emb_name, methods in importance_matrices.items():
        for method, mat in methods.items():
            for t_idx, ds_name in enumerate(task_names):
                records.append({
                    "embedding": emb_name,
                    "method":    method.upper(),
                    "dataset":   ds_name,
                    "n_nonzero": (mat[:, t_idx] > 1e-8).sum(),
                })
    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, method in zip(axes, ["LASSO", "ENET"]):
        sns.barplot(data=df[df["method"] == method], x="dataset", y="n_nonzero",
                    hue="embedding", ax=ax, palette="tab10")
        ax.set_title(f"{method} — Non-zero Dims per Dataset", fontsize=12)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("# Non-zero Dimensions")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Embedding")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_stability_heatmap(stability_freqs: dict, task_names: list,
                            top_k: int = 64, out_path: Path = None):
    """Heatmap of bootstrap selection frequencies (rows=top dims, cols=tasks)."""
    fig, axes = plt.subplots(1, len(stability_freqs), figsize=(7 * len(stability_freqs), 14))
    if len(stability_freqs) == 1:
        axes = [axes]

    for ax, (emb_name, mat) in zip(axes, stability_freqs.items()):
        top_idx = np.argsort(mat.max(axis=1))[::-1][:top_k]
        df_plot = pd.DataFrame(mat[top_idx], index=[f"dim {i}" for i in top_idx], columns=task_names)
        sns.heatmap(df_plot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
                    cmap="Blues", linewidths=0.3, cbar_kws={"label": "Selection Probability"})
        ax.set_title(f"{emb_name} — Stability Selection\n(top {top_k} dims by max freq)", fontsize=12)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_mi_heatmap(mi_matrices: dict, task_names: list,
                    top_k: int = 64, out_path: Path = None):
    """Heatmap of normalized mutual information per dim (rows=top dims, cols=tasks)."""
    fig, axes = plt.subplots(1, len(mi_matrices), figsize=(8 * len(mi_matrices), 14))
    if len(mi_matrices) == 1:
        axes = [axes]

    for ax, (emb_name, mat) in zip(axes, mi_matrices.items()):
        col_max = mat.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        mat_norm = mat / col_max
        top_idx = np.argsort(mat_norm.mean(axis=1))[::-1][:top_k]
        df_plot = pd.DataFrame(mat_norm[top_idx], index=[f"dim {i}" for i in top_idx], columns=task_names)
        sns.heatmap(df_plot, ax=ax, cmap="viridis", vmin=0, vmax=1, linewidths=0,
                    cbar_kws={"label": "Normalized MI (per task)"})
        ax.set_title(f"{emb_name} — Mutual Information\n(top {top_k} dims)", fontsize=12)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_mrmr_heatmap(mrmr_results: dict, embeddings: dict, task_names: list,
                       n_mrmr: int = 32, out_path: Path = None):
    """Heatmap of mRMR selection rank per dim (low rank = selected early = more informative)."""
    fig, axes = plt.subplots(1, len(embeddings), figsize=(8 * len(embeddings), 12))
    if len(embeddings) == 1:
        axes = [axes]

    cmap = plt.cm.get_cmap("RdYlGn").reversed()

    for ax, emb_name in zip(axes, embeddings):
        n_dims = embeddings[emb_name].shape[1]
        rank_mat = np.full((n_dims, len(task_names)), np.nan)
        for t_idx, ds_name in enumerate(task_names):
            for rank, dim_idx in enumerate(mrmr_results[emb_name][ds_name], start=1):
                rank_mat[dim_idx, t_idx] = rank

        selected_rows = np.where(~np.all(np.isnan(rank_mat), axis=1))[0]
        df_plot = pd.DataFrame(rank_mat[selected_rows],
                               index=[f"dim {i}" for i in selected_rows],
                               columns=task_names)
        sns.heatmap(df_plot, ax=ax, cmap=cmap, vmin=1, vmax=n_mrmr,
                    linewidths=0.2, cbar_kws={"label": "Selection Rank (1=first)"})
        ax.set_title(f"{emb_name} — mRMR Selection Rank\n(dims selected in ≥1 task)", fontsize=12)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()


def plot_ablation_curves(ablation_curves: dict, ablation_task_types: dict,
                          embeddings: dict, out_path: Path = None):
    """Line plots of performance vs. fraction of top dims removed."""
    task_names = list(ablation_curves.keys())
    n_tasks = len(task_names)
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, n_tasks, figsize=(5 * n_tasks, 4))
    if n_tasks == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, task_names):
        task_type = ablation_task_types[ds_name]
        metric_label = "R²" if task_type == "regression" else "Accuracy"
        for i, emb_name in enumerate(embeddings):
            fracs, scores = ablation_curves[ds_name][emb_name]
            ax.plot(fracs * 100, scores, marker="o", label=emb_name, color=colors[i])
        ax.set_title(ds_name, fontsize=12)
        ax.set_xlabel("% Most-Important Dims Removed")
        ax.set_ylabel(metric_label)
        ax.legend(title="Embedding", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Progressive Ablation — Sharp drop = concentrated signal; Gradual = distributed",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()

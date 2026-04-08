"""
Plotting utilities for feature selection analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def _adaptive_fig_height(n_rows, min_height=6, per_row=0.04, max_height=40):
    return min(max(min_height, n_rows * per_row), max_height)


def _sparse_yticks(ax, n_rows, target_ticks=20):
    stride = max(1, n_rows // target_ticks)
    ticks = np.arange(0, n_rows, stride)
    ax.set_yticks(ticks + 0.5)
    ax.set_yticklabels(ticks, rotation=0)


def plot_importance_heatmap(
    importance_matrices: dict,
    task_names: list,
    method: str = "lasso",
    out_path: Path = None,
):
    """One heatmap PNG per embedding, height proportional to dimensionality."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 11,
        "axes.titlepad": 10,
    })
    sns.set_style("white")

    n_tasks = len(task_names)
    fig_width = max(6, 1.6 * n_tasks)   # ~1.6 in per task column

    for emb_name in importance_matrices:
        mat = importance_matrices[emb_name][method]  # (n_dims, n_tasks)
        n_dims = mat.shape[0]

        height_per_dim = 0.025
        fig_height = max(6, n_dims * height_per_dim)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        df_plot = pd.DataFrame(
            mat,
            index=np.arange(n_dims),
            columns=task_names,
        )
        sns.heatmap(
            df_plot,
            ax=ax,
            cmap="magma",
            vmin=0, vmax=1,
            linewidths=0,
            cbar_kws={"label": f"|{method.upper()}| (max-norm)", "shrink": 0.6},
        )

        ax.set_title(emb_name, fontweight="bold")
        ax.set_xlabel("Dataset", labelpad=8)
        ax.set_ylabel("Embedding Dimension", labelpad=8)
        ax.tick_params(axis="x", rotation=45)

        # ~20 evenly-spaced y-tick labels at cell centres (i + 0.5)
        stride = max(1, n_dims // 20)
        tick_rows = np.arange(0, n_dims, stride)
        ax.set_yticks(tick_rows + 0.5)
        ax.set_yticklabels(tick_rows, rotation=0)

        # seaborn sets ylim correctly; do not override

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            save_path = out_path.with_name(f"{stem}_{emb_name}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close(fig)


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

    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(8 * n_methods, 5), sharey=False)
    if n_methods == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
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
    plt.close()


def plot_stability_heatmap(stability_freqs: dict, task_names: list,
                          out_path: Path = None):

    for emb_name, mat in stability_freqs.items():
        n_dims = mat.shape[0]

        fig_height = _adaptive_fig_height(n_dims)
        fig_width = max(6, 1.6 * len(task_names))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        df_plot = pd.DataFrame(
            mat,
            index=np.arange(n_dims),
            columns=task_names
        )

        sns.heatmap(
            df_plot,
            ax=ax,
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0,
            cbar_kws={"label": "Selection Probability", "shrink": 0.6},
        )

        ax.set_title(f"{emb_name} — Stability Selection", fontsize=14)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

        _sparse_yticks(ax, n_dims)

        if out_path:
            save_path = Path(out_path).with_name(f"{Path(out_path).stem}_{emb_name}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close()

def plot_mi_heatmap(mi_matrices: dict, task_names: list,
                    out_path: Path = None):

    for emb_name, mat in mi_matrices.items():
        n_dims = mat.shape[0]

        col_max = mat.max(axis=0, keepdims=True)  # (1, n_tasks)
        safe_max = np.where(col_max > 0, col_max, 1)
        mat_norm = np.where(col_max > 0, mat / safe_max, np.nan)

        fig_height = _adaptive_fig_height(n_dims)
        fig_width = max(6, 1.6 * len(task_names))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        df_plot = pd.DataFrame(
            mat_norm,
            index=np.arange(n_dims),
            columns=task_names
        )

        sns.heatmap(
            df_plot,
            ax=ax,
            cmap="viridis",
            vmin=0,
            vmax=1,
            linewidths=0,
            cbar_kws={"label": "Normalized MI", "shrink": 0.6},
        )

        ax.set_title(f"{emb_name} — Mutual Information", fontsize=14)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

        _sparse_yticks(ax, n_dims)

        if out_path:
            save_path = Path(out_path).with_name(f"{Path(out_path).stem}_{emb_name}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close()


def plot_mrmr_heatmap(mrmr_results: dict, task_names: list,
                     n_mrmr: int = 32, out_path: Path = None):
    cmap = plt.cm.get_cmap("RdYlGn").reversed()

    for emb_name in mrmr_results:
        flat = [i for ds in task_names for i in mrmr_results[emb_name].get(ds, [])]
        n_dims = max(flat) + 1 if flat else 1
        rank_mat = np.full((n_dims, len(task_names)), np.nan)

        for t_idx, ds_name in enumerate(task_names):
            for rank, dim_idx in enumerate(mrmr_results[emb_name][ds_name], start=1):
                rank_mat[dim_idx, t_idx] = rank

        fig_height = _adaptive_fig_height(n_dims)
        fig_width = max(6, 1.6 * len(task_names))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

        df_plot = pd.DataFrame(
            rank_mat,
            index=np.arange(n_dims),
            columns=task_names
        )

        sns.heatmap(
            df_plot,
            ax=ax,
            cmap=cmap,
            vmin=1,
            vmax=n_mrmr,
            linewidths=0,
            mask=np.isnan(df_plot),
            cbar_kws={"label": "Selection Rank (1=first)", "shrink": 0.6},
        )

        ax.set_title(f"{emb_name} — mRMR Rank", fontsize=14)
        ax.set_xlabel("Task")
        ax.set_ylabel("Embedding Dimension")
        ax.tick_params(axis="x", rotation=45)

        _sparse_yticks(ax, n_dims)

        if out_path:
            save_path = Path(out_path).with_name(f"{Path(out_path).stem}_{emb_name}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close()


def plot_ablation_curves(ablation_curves: dict, ablation_task_types: dict,
                          embedding_names: list, out_path: Path = None):
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
        for i, emb_name in enumerate(embedding_names):
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

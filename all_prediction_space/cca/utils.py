"""
Plotting utilities for embedding–target linear association (CCA or scalar-Y linear scores).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def _adaptive_fig_height(n_rows, min_height=6, per_row=0.025, max_height=40):
    return min(max(min_height, n_rows * per_row), max_height)


def _sparse_yticks(ax, n_rows, target_ticks=20):
    stride = max(1, n_rows // target_ticks)
    ticks = np.arange(0, n_rows, stride)
    ax.set_yticks(ticks + 0.5)
    ax.set_yticklabels(ticks, rotation=0)


# ---------------------------------------------------------------------------
# Heatmap: all embedding dims × all tasks (overview)
# ---------------------------------------------------------------------------

def plot_cca_weight_heatmap(
    cca_weights: dict,
    task_names: list,
    component: int = 0,
    out_path: Path = None,
    *,
    method_label: str = "CCA",
    loading_bar_label: str | None = None,
):
    """
    Heatmap of the nth loading vector for each embedding (CCA or scalar-Y linear score).

    cca_weights[emb_name][ds_name] = loading array (n_dims, n_components)
    """
    bar_lbl = loading_bar_label or f"{method_label} loading"
    sns.set_style("white")

    for emb_name, task_dict in cca_weights.items():
        n_dims = next(iter(task_dict.values())).shape[0]
        mat = np.full((n_dims, len(task_names)), np.nan)
        for t_idx, ds_name in enumerate(task_names):
            w = task_dict.get(ds_name)
            if w is not None and w.shape[1] > component:
                mat[:, t_idx] = w[:, component]

        # Normalise each column to [-1, 1] by max absolute value; blank if unavailable
        col_absmax = np.nanmax(np.abs(mat), axis=0, keepdims=True)
        safe_absmax = np.where(col_absmax > 0, col_absmax, 1)
        mat_norm = np.where(col_absmax > 0, mat / safe_absmax, np.nan)

        fig_height = _adaptive_fig_height(n_dims)
        fig_width = max(6, 1.6 * len(task_names))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        df_plot = pd.DataFrame(mat_norm, index=np.arange(n_dims), columns=task_names)
        unavailable_mask = np.isnan(df_plot)

        ax.set_facecolor("#cccccc")
        sns.heatmap(
            df_plot, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0,
            mask=unavailable_mask,
            cbar_kws={"label": f"{bar_lbl} (comp {component + 1}, abs-norm)", "shrink": 0.6},
        )

        unavail_cols = [task_names[j] for j in range(len(task_names))
                        if unavailable_mask.values[:, j].all()]
        note = f"\n(gray = unavailable: {', '.join(unavail_cols)})" if unavail_cols else ""
        ax.set_title(f"{emb_name} — {method_label} loadings (component {component + 1}){note}",
                     fontweight="bold", fontsize=14)
        ax.set_xlabel("Dataset", labelpad=8)
        ax.set_ylabel("Embedding Dimension", labelpad=8)
        ax.tick_params(axis="x", rotation=45)
        _sparse_yticks(ax, n_dims)

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            save_path = out_path.with_name(f"{stem}_{emb_name}_comp{component + 1}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# View 1 — variable-centric: for each task, which dims are most important?
# ---------------------------------------------------------------------------

def plot_variable_loadings(
    cca_weights: dict,
    task_names: list,
    top_k: int = 30,
    component: int = 0,
    out_path: Path = None,
    *,
    method_label: str = "CCA",
    projection_xlabel: str | None = None,
):
    """
    For each environmental variable, plot a horizontal bar chart of the top-k
    embedding dimensions by absolute projection weight on the first component.

    cca_weights[emb_name][ds_name] = weights array (n_dims, n_components)
    """
    px_label = projection_xlabel or f"{method_label} projection weight"
    emb_names = list(cca_weights.keys())
    n_embs = len(emb_names)

    for ds_name in task_names:
        fig, axes = plt.subplots(1, n_embs, figsize=(6 * n_embs, max(4, top_k * 0.28)),
                                 sharey=False)
        if n_embs == 1:
            axes = [axes]

        for ax, emb_name in zip(axes, emb_names):
            weights = cca_weights[emb_name].get(ds_name)
            if weights is None or weights.shape[1] <= component:
                ax.set_title(f"{emb_name}\n(unavailable)", fontsize=11)
                ax.axis("off")
                continue

            vec = weights[:, component]           # (n_dims,)
            top_idx = np.argsort(np.abs(vec))[-top_k:][::-1]
            vals  = vec[top_idx]
            labels = [f"dim {i}" for i in top_idx]
            colors = ["#d73027" if v > 0 else "#4575b4" for v in vals]

            ax.barh(np.arange(len(vals))[::-1], vals, color=colors)
            ax.set_yticks(np.arange(len(vals))[::-1])
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel(px_label)
            ax.set_title(f"{emb_name}", fontsize=12)

        fig.suptitle(
            f"Variable-centric: {ds_name}  (component {component + 1})\n"
            f"Top-{top_k} embedding dims by |{method_label} weight|",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            safe_name = ds_name.replace(" ", "_").lower()
            save_path = out_path.with_name(f"{stem}_{safe_name}_comp{component + 1}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# View 2 — dimension-centric: for each dim, which env vars correlate with it?
# ---------------------------------------------------------------------------

def plot_dim_label_correlations(
    corr_matrices: dict,
    task_names: list,
    out_path: Path = None,
):
    """
    Heatmap of Pearson r between each embedding dimension and each regression label.
    Answers: "For each embedding dim — which env vars does it track?"

    corr_matrices[emb_name] = (n_dims, n_tasks) array; NaN for classification tasks.
    """
    sns.set_style("white")
    regression_cols = [j for j, ds in enumerate(task_names)
                       if not np.all(np.isnan(next(iter(corr_matrices.values()))[:, j]))]
    reg_names = [task_names[j] for j in regression_cols]

    for emb_name, mat in corr_matrices.items():
        mat_reg = mat[:, regression_cols]          # (n_dims, n_reg_tasks)
        n_dims  = mat_reg.shape[0]

        fig_height = _adaptive_fig_height(n_dims)
        fig_width  = max(4, 1.6 * len(reg_names))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        df_plot = pd.DataFrame(mat_reg, index=np.arange(n_dims), columns=reg_names)

        ax.set_facecolor("#cccccc")
        sns.heatmap(
            df_plot, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0,
            mask=np.isnan(df_plot),
            cbar_kws={"label": "Pearson r", "shrink": 0.6},
        )

        ax.set_title(f"{emb_name} — Dim–Label Correlations",
                     fontweight="bold", fontsize=14)
        ax.set_xlabel("Environmental Variable", labelpad=8)
        ax.set_ylabel("Embedding Dimension", labelpad=8)
        ax.tick_params(axis="x", rotation=45)
        _sparse_yticks(ax, n_dims)

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            save_path = out_path.with_name(f"{stem}_{emb_name}.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# View 2 — dimension-centric CCA: for each dim, which env-var combination aligns?
# ---------------------------------------------------------------------------

def plot_vars_to_dim_weights(
    vars_to_dim_weights: dict,
    vars_to_dim_corrs: dict,
    reg_task_names: list,
    top_k: int = 30,
    out_path: Path = None,
):
    """
    For each embedding, show which linear combination of environmental variables
    is most aligned with each embedding dimension (CCA projection weights on env-vars side).

    Displays the top-K embedding dims (ranked by canonical correlation), each as a
    horizontal bar chart of env-var weights.  Also produces a heatmap overview.

    Answers: "For embedding dim k — which combination of env vars aligns with it?"

    vars_to_dim_weights[emb_name] : (n_dims, n_env_vars)
    vars_to_dim_corrs[emb_name]   : (n_dims,)
    """
    sns.set_style("white")
    n_vars = len(reg_task_names)

    for emb_name in vars_to_dim_weights:
        weights_mat = vars_to_dim_weights[emb_name]   # (n_dims, n_env_vars)
        corrs_arr   = vars_to_dim_corrs[emb_name]     # (n_dims,)
        n_dims = weights_mat.shape[0]

        # --- Heatmap overview: top-K dims (by canonical correlation) ---
        top_dim_idx = np.argsort(np.abs(corrs_arr))[-top_k:][::-1]
        mat_sub  = weights_mat[top_dim_idx]            # (top_k, n_env_vars)
        corr_sub = corrs_arr[top_dim_idx]

        # Normalise rows to [-1, 1] for visual comparison
        row_absmax = np.abs(mat_sub).max(axis=1, keepdims=True)
        safe_max   = np.where(row_absmax > 0, row_absmax, 1)
        mat_norm   = mat_sub / safe_max

        row_labels = [f"dim {i}  (r={corrs_arr[i]:.2f})" for i in top_dim_idx]

        fig_h = max(6, top_k * 0.35)
        fig_w = max(5, 1.4 * n_vars)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        df_plot = pd.DataFrame(mat_norm, index=row_labels, columns=reg_task_names)
        sns.heatmap(
            df_plot, ax=ax,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.3,
            cbar_kws={"label": "CCA Weight (row-normalised)", "shrink": 0.6},
        )
        ax.set_title(
            f"{emb_name} — Dim-centric CCA\n"
            f"Top-{top_k} dims: env-var projection weights",
            fontweight="bold", fontsize=13,
        )
        ax.set_xlabel("Environmental Variable", labelpad=8)
        ax.set_ylabel("Embedding Dimension", labelpad=8)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", labelsize=8)

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            hmap_path = out_path.with_name(f"{stem}_{emb_name}_heatmap.png")
            plt.savefig(hmap_path, dpi=150, bbox_inches="tight")
            print(f"Saved {hmap_path}")

        plt.show()
        plt.close(fig)

        # --- Per-dim bar charts for the same top-K dims ---
        ncols = min(4, top_k)
        nrows = int(np.ceil(top_k / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.5 * nrows),
                                 constrained_layout=True)
        axes_flat = np.array(axes).flatten()

        for ax_i, dim_i in enumerate(top_dim_idx):
            ax = axes_flat[ax_i]
            w = weights_mat[dim_i]
            colors = ["#d73027" if v > 0 else "#4575b4" for v in w]
            ax.barh(np.arange(n_vars)[::-1], w[::-1], color=colors[::-1])
            ax.set_yticks(np.arange(n_vars))
            ax.set_yticklabels(reg_task_names[::-1], fontsize=7)
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set_title(f"dim {dim_i}  r={corrs_arr[dim_i]:.2f}", fontsize=9)
            ax.set_xlabel("CCA Weight", fontsize=8)

        for ax in axes_flat[top_k:]:
            ax.axis("off")

        fig.suptitle(
            f"{emb_name} — Dim-centric CCA: env-var projection weights\n"
            f"Top-{top_k} dims by |canonical correlation|",
            fontsize=12, fontweight="bold",
        )

        if out_path:
            out_path = Path(out_path)
            bars_path = out_path.with_name(f"{stem}_{emb_name}_bars.png")
            plt.savefig(bars_path, dpi=150, bbox_inches="tight")
            print(f"Saved {bars_path}")

        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Clustered heatmap: hierarchical clustering on embedding dims × tasks
# ---------------------------------------------------------------------------

def plot_cca_weight_heatmap_clustered(
    cca_weights: dict,
    task_names: list,
    component: int = 0,
    out_path: Path = None,
    *,
    method_label: str = "CCA",
    abs: bool = True,
    loading_bar_label: str | None = None,
    top_k: int = 512,
    row_cluster: bool = True,
    col_cluster: bool = True,
    linkage_method: str = "ward",
    metric: str = "euclidean",
):
    """
    Clustered heatmap of the nth loading vector for each embedding.

    Uses sns.clustermap to apply hierarchical clustering on rows (embedding
    dimensions) and/or columns (tasks), revealing groups of dims that respond
    similarly across tasks.

    top_k: keep only the top-k rows by row L2 norm. Plotting all 512+ dims
    produces rows too thin to see — top_k makes the dendrogram readable.

    cca_weights[emb_name][ds_name] = loading array (n_dims, n_components)
    """
    bar_lbl = loading_bar_label or f"{method_label} loading"
    sns.set_style("white")

    for emb_name, task_dict in cca_weights.items():
        n_dims = next(iter(task_dict.values())).shape[0]
        mat = np.full((n_dims, len(task_names)), np.nan)
        for t_idx, ds_name in enumerate(task_names):
            w = task_dict.get(ds_name)
            if w is not None and w.shape[1] > component:
                mat[:, t_idx] = np.abs(w[:, component])

        # Normalise each column to [-1, 1] by max absolute value
        col_absmax = np.nanmax(np.abs(mat), axis=0, keepdims=True)
        safe_absmax = np.where(col_absmax > 0, col_absmax, 1)
        mat_norm = np.where(col_absmax > 0, mat / safe_absmax, 0.0)

        # Keep top-k rows by L2 norm so dendrograms are visible
        row_norms = np.linalg.norm(mat_norm, axis=1)
        k = min(top_k, n_dims)
        top_idx = np.argsort(row_norms)[-k:]
        mat_sub = mat_norm[top_idx]
        row_labels = [f"dim {i}" for i in top_idx]

        df_plot = pd.DataFrame(mat_sub, index=row_labels, columns=task_names)

        fig_h = max(8, k * 0.28)
        fig_w = max(6, 1.6 * len(task_names))

        vmin = 0 if abs else np.nanmin(mat_sub)
        vmax = 1
        cmap = "Reds" if abs else "RdBu_r"
        center = None if abs else 0

        g = sns.clustermap(
            df_plot,
            cmap=cmap, #only one color if abs
            center=center, vmin=vmin, vmax=vmax,
            linewidths=0.3,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            method=linkage_method,
            metric=metric,
            yticklabels=True,
            xticklabels=True,
            figsize=(fig_w, fig_h),
            cbar_kws={"label": f"{bar_lbl} (comp {component + 1}, abs-norm)", "shrink": 0.6},
            dendrogram_ratio=(0.2, 0.1),
        )

        # Reposition colorbar to the right of the heatmap (clustermap defaults
        # to top-left, which overlaps the column dendrogram).
        g.figure.canvas.draw()
        hm_pos = g.ax_heatmap.get_position()
        g.cax.set_position([
            hm_pos.x1 + 0.09,                  # just right of heatmap
            hm_pos.y0 + hm_pos.height * 0.2,   # vertically centred
            0.02,
            hm_pos.height * 0.6,
        ])

        g.ax_heatmap.set_xlabel("Dataset", labelpad=8)
        g.ax_heatmap.set_ylabel("Embedding Dimension", labelpad=8)
        g.ax_heatmap.tick_params(axis="x", rotation=45)
        g.ax_heatmap.tick_params(axis="y", rotation=0, labelsize=7)
        g.figure.suptitle(
            f"{emb_name} — {method_label} loadings (component {component + 1})\n"
            f"Top-{k} dims by L2 norm · clustering: method={linkage_method}, metric={metric}",
            fontweight="bold", fontsize=13, y=1.04,
        )

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            save_path = out_path.with_name(f"{stem}_{emb_name}_comp{component + 1}_clustered.png")
            g.figure.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"Saved {save_path}")

        plt.show()
        plt.close(g.figure)


# ---------------------------------------------------------------------------
# Canonical correlations bar chart
# ---------------------------------------------------------------------------

def plot_canonical_correlations(
    cca_corrs: dict,
    task_names: list,
    n_components: int = 1,
    out_path: Path = None,
    *,
    method_label: str = "CCA",
    score_label: str = "Canonical correlation (r)",
):
    """
    Grouped bar chart of correlation scores per component.
    With scalar Y and one component, values are the multiple correlation R.

    cca_corrs[emb_name][ds_name] = corrs array (n_components,)
    """
    records = []
    for emb_name, task_dict in cca_corrs.items():
        for ds_name in task_names:
            corrs = task_dict.get(ds_name)
            if corrs is None:
                continue
            for comp in range(min(n_components, len(corrs))):
                records.append({
                    "embedding": emb_name,
                    "dataset":   ds_name,
                    "component": f"comp {comp + 1}",
                    "r":         float(corrs[comp]),
                })
    df = pd.DataFrame(records)
    n_comp = df["component"].nunique()

    fig, axes = plt.subplots(1, n_comp, figsize=(7 * n_comp, 5), sharey=False)
    if n_comp == 1:
        axes = [axes]

    for ax, comp_label in zip(axes, sorted(df["component"].unique())):
        sub = df[df["component"] == comp_label]
        sns.barplot(data=sub, x="dataset", y="r", hue="embedding", ax=ax, palette="tab10")
        ax.set_title(f"{method_label} — {comp_label}", fontsize=12)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(score_label)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Embedding")
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved {out_path}")
    plt.show()

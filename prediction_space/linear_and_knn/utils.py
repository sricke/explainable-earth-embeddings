import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_TASK_LABELS = {
    'air_temp': 'Air Temp', 'elevation': 'Elevation',
    'nightlights': 'Nightlights', 'pop_density': 'Pop Density',
    'treecover': 'Treecover', 'biome': 'Biome',
    'ecoregion': 'Ecoregion', 'countries': 'Countries',
}


def plot_linear_results(df: pd.DataFrame, out_path=None):
    """Grouped bar chart of Ridge R² (regression) and accuracy (classification)."""
    reg_df = df[df['metric'] == 'r2'].copy()
    cls_df = df[df['metric'] == 'accuracy'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    specs = [
        (reg_df,  'Ridge Regression — R²',         'R²'),
        (cls_df,  'Ridge Classifier — Accuracy', 'Accuracy'),
    ]
    for ax, (data, title, ylabel) in zip(axes, specs):
        if data.empty:
            ax.set_visible(False)
            continue
        data = data.copy()
        data['task_label'] = data['task'].map(_TASK_LABELS).fillna(data['task'])
        sns.barplot(data=data, x='task_label', y='value', hue='backend',
                    ax=ax, palette='tab10')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Dataset')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title='Backend')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()
    plt.close()


def plot_linear_heatmap(df: pd.DataFrame, out_path=None):
    """Heatmap of Ridge R² and Accuracy: tasks × backends."""
    specs = [
        ('r2',       'Ridge Regression — R²',         'RdYlGn'),
        ('accuracy', 'Ridge Classifier — Accuracy', 'RdYlGn'),
    ]
    panels = [(m, t, c) for m, t, c in specs if not df[df['metric'] == m].empty]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)

    for ax, (metric, title, cmap) in zip(axes[0], panels):
        sub = df[df['metric'] == metric].copy()
        sub['task_label'] = sub['task'].map(_TASK_LABELS).fillna(sub['task'])
        pivot = sub.pivot(index='task_label', columns='backend', values='value')

        sns.heatmap(
            pivot, ax=ax, cmap=cmap, vmin=0, vmax=1, annot=True, fmt='.2f',
            linewidths=0.5, cbar_kws={'label': title.split('—')[1].strip(), 'shrink': 0.8},
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Backend')
        ax.set_ylabel('Dataset')
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()
    plt.close()


def plot_knn_heatmap(knn_df: pd.DataFrame, k: int = 5, out_path=None):
    """Heatmap of KNN R² and Accuracy at a fixed k: tasks × backends."""
    sub_k = knn_df[knn_df['k'] == k]
    if sub_k.empty:
        print(f"No KNN results for k={k}")
        return

    specs = [
        ('r2',       f'KNN Regression — R² (k={k})',         'RdYlGn'),
        ('accuracy', f'KNN Classifier — Accuracy (k={k})', 'RdYlGn'),
    ]
    panels = [(m, t, c) for m, t, c in specs if not sub_k[sub_k['metric'] == m].empty]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)

    for ax, (metric, title, cmap) in zip(axes[0], panels):
        sub = sub_k[sub_k['metric'] == metric].copy()
        sub['task_label'] = sub['task'].map(_TASK_LABELS).fillna(sub['task'])
        pivot = sub.pivot(index='task_label', columns='backend', values='value')

        sns.heatmap(
            pivot, ax=ax, cmap=cmap, vmin=0, vmax=1, annot=True, fmt='.2f',
            linewidths=0.5, cbar_kws={'label': title.split('—')[1].strip(), 'shrink': 0.8},
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Backend')
        ax.set_ylabel('Dataset')
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()
    plt.close()


def plot_knn_results(knn_df: pd.DataFrame, out_path=None):
    """Line plots of metric vs k, one subplot per task."""
    tasks = list(knn_df['task'].unique())
    backends = list(knn_df['backend'].unique())
    n_tasks = len(tasks)
    n_cols = min(4, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for ax, task in zip(axes_flat, tasks):
        task_data = knn_df[knn_df['task'] == task]
        metric = task_data['metric'].iloc[0]
        ylabel = 'R²' if metric == 'r2' else 'Accuracy'

        for i, backend in enumerate(backends):
            bd = task_data[task_data['backend'] == backend].sort_values('k')
            ax.plot(bd['k'], bd['value'], marker='o', label=backend,
                    color=colors[i % len(colors)])

        ax.set_title(_TASK_LABELS.get(task, task), fontsize=11)
        ax.set_xlabel('k')
        ax.set_ylabel(ylabel)
        ax.legend(title='Backend', fontsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n_tasks:]:
        ax.set_visible(False)

    fig.suptitle('KNN Performance vs. k', fontsize=14, y=1.01)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved {out_path}")
    plt.show()
    plt.close()

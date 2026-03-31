import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

sys.path.append("..")
sys.path.append("../..")
from utils import load_embeddings, load_labels, REGRESSION_DATASETS, CLASSIFICATION_DATASETS
from probes import probe_knn_regression, probe_knn_classification, K_VALUES
from gpu_probes import probe_knn_regression_gpu, probe_knn_classification_gpu, K_VALUES

FIGURES_DIR = Path("../../figures")


def run_probes(embeddings, regression_labels, classification_labels, use_gpu=False):
    probe_knn_regression_fn = probe_knn_regression_gpu if use_gpu else probe_knn_regression
    probe_knn_classification_fn = probe_knn_classification_gpu if use_gpu else probe_knn_classification
    results = []

    for emb_name, X in embeddings.items():
        print(f"Probing {emb_name}...")

        for ds_name, y in tqdm(regression_labels.items(), desc="Regression Datasets"):
            scores = probe_knn_regression_fn(X, y)
            for k, (mean, std) in scores.items():
                results.append({"embedding": emb_name, "dataset": ds_name,
                                 "task": "regression", "k": k, "score": mean, "std": std})
            summary = ", ".join(f"k={k}: {m:.4f}±{s:.4f}" for k, (m, s) in scores.items())
            print(f"  {ds_name}: {summary}")

        for ds_name, y in tqdm(classification_labels.items(), desc="Classification Datasets"):
            scores = probe_knn_classification_fn(X, y)
            for k, (mean, std) in scores.items():
                results.append({"embedding": emb_name, "dataset": ds_name,
                                 "task": "classification", "k": k, "score": mean, "std": std})
            summary = ", ".join(f"k={k}: {m:.4f}±{s:.4f}" for k, (m, s) in scores.items())
            print(f"  {ds_name}: {summary}")

    return pd.DataFrame(results)


def plot_heatmaps(df, task, dataset_names, embedding_names, metric_label, out_path):
    k_methods = K_VALUES
    fig, axes = plt.subplots(1, len(k_methods), figsize=(6 * len(k_methods), max(3, len(dataset_names))))
    for ax, k in zip(axes, k_methods):
        pivot = (
            df[(df["task"] == task) & (df["k"] == k)]
            .pivot(index="dataset", columns="embedding", values="score")
            .reindex(index=dataset_names, columns=embedding_names)
        )
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
                    cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric_label})
        ax.set_title(f"kNN-{k} {metric_label}", fontsize=13)
        ax.set_xlabel("Embedding")
        ax.set_ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_k_curves(df, task, metric_label, out_path):
    datasets = df[df["task"] == task]["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(4 * n_ds, 4), sharey=False)
    if n_ds == 1:
        axes = [axes]
    for ax, ds_name in zip(axes, datasets):
        sub = df[(df["task"] == task) & (df["dataset"] == ds_name)]
        for emb_name, grp in sub.groupby("embedding"):
            grp = grp.sort_values("k")
            ax.plot(grp["k"], grp["score"], marker="o", label=emb_name)
        ax.set_title(ds_name, fontsize=11)
        ax.set_xlabel("k")
        ax.set_ylabel(metric_label)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(f"kNN {task.title()} — Score vs. k", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading embeddings and labels...")
    latlons, embeddings = load_embeddings()
    regression_labels, classification_labels = load_labels(latlons)

    for name, emb in embeddings.items():
        print(f"  {name}: shape={emb.shape}, dtype={emb.dtype}")

    print("\nRunning probes...")
    df = run_probes(embeddings, regression_labels, classification_labels, use_gpu=True)
    df.to_csv(f"{FIGURES_DIR}/knn_probing_results.csv")
    print(df.to_string())

    embedding_names = list(embeddings.keys())

    plot_heatmaps(df, "regression", list(REGRESSION_DATASETS.keys()), embedding_names,
                  "R²", FIGURES_DIR / "knn" / "regression" / "knn_regression_heatmap.png")
    plot_heatmaps(df, "classification", list(CLASSIFICATION_DATASETS.keys()), embedding_names,
                  "Accuracy", FIGURES_DIR / "knn" / "classification" / "knn_classification_heatmap.png")
    plot_k_curves(df, "regression", "R²", FIGURES_DIR / "knn" / "regression" / "knn_regression_k_curve.png")
    plot_k_curves(df, "classification", "Accuracy", FIGURES_DIR / "knn" / "classification" / "knn_classification_k_curve.png")

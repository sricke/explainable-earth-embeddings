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
from probes import probe_ridge, probe_logistic
from gpu_probes import probe_ridge_gpu, probe_logistic_gpu
from utils import load_embeddings, load_labels, REGRESSION_DATASETS, CLASSIFICATION_DATASETS

FIGURES_DIR = Path("../../figures")


def run_probes(embeddings, regression_labels, classification_labels, use_gpu=False):
    probe_ridge_fn = probe_ridge_gpu if use_gpu else probe_ridge
    probe_logistic_fn = probe_logistic_gpu if use_gpu else probe_logistic
    results = []

    for emb_name, X in embeddings.items():
        print(f'Probing {emb_name}...')
        for ds_name, y in tqdm(regression_labels.items(), desc='Regression Datasets'):
            score_mean, score_std = probe_ridge_fn(X, y)
            results.append({
                'embedding': emb_name, 
                'dataset': ds_name, 
                'task': 'regression', 
                'score': score_mean,
                'std': score_std
            })
            
            print(f'  {ds_name}: R²={score_mean:.4f} ± {score_std:.4f}')

        for ds_name, y in tqdm(classification_labels.items(), desc='Classification Datasets'):
            score_mean, score_std = probe_logistic_fn(X, y)
            results.append({
                'embedding': emb_name, 
                'dataset': ds_name, 
                'task': 'classification', 
                'score': score_mean,
                'std': score_std
                })
            print(f'  {ds_name}: Accuracy={score_mean:.4f} ± {score_std:.4f}')

    df = pd.DataFrame(results)
    return df


def plot_heatmap(df, task, dataset_names, embedding_names, metric_label, out_path):
    sub = df[df["task"] == task]
    pivot = sub.pivot(index="dataset", columns="embedding", values="score").reindex(
        index=dataset_names, columns=embedding_names
    )
    fig, ax = plt.subplots(figsize=(7, max(3, len(dataset_names))))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric_label})
    ax.set_title(f"{'Ridge Regression' if task == 'regression' else 'Logistic Regression'} {metric_label}", fontsize=14)
    ax.set_xlabel("Embedding")
    ax.set_ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_bars(df, task, metric_label, out_path):
    sub = df[df["task"] == task]
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=sub, x="dataset", y="score", hue="embedding", ax=ax, palette="tab10")
    ax.set_title(f"{task.title()} — Ridge/Logistic {metric_label} by Dataset", fontsize=13)
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Dataset")
    ax.set_ylim(0, 1)
    ax.legend(title="Embedding")
    plt.xticks(rotation=20, ha="right")
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
    df.to_csv(f"{FIGURES_DIR}/ridge_probing_results.csv")
    print(df.to_string())

    embedding_names = list(embeddings.keys())

    plot_heatmap(df, "regression", list(REGRESSION_DATASETS.keys()), embedding_names,
                 "R²", FIGURES_DIR / "ridge" / "regression" / "ridge_regression_heatmap.png")
    plot_heatmap(df, "classification", list(CLASSIFICATION_DATASETS.keys()), embedding_names,
                 "Accuracy", FIGURES_DIR / "ridge" / "classification" / "ridge_classification_heatmap.png")
    plot_bars(df, "regression", "R²", FIGURES_DIR / "ridge" / "regression" / "ridge_regression_bars.png")
    plot_bars(df, "classification", "Accuracy", FIGURES_DIR / "ridge" / "classification" / "ridge_classification_bars.png")

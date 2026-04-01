import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

warnings.filterwarnings("ignore")

sys.path.append("..")
sys.path.append("../..")
from utils import load_embeddings_and_labels, EMBEDDING_BACKENDS, REGRESSION_DATASETS, CLASSIFICATION_DATASETS
from probes import probe_knn_regression, probe_knn_classification, K_VALUES
from gpu_probes import probe_knn_regression_gpu, probe_knn_classification_gpu, K_VALUES

FIGURES_DIR = Path("../../figures")
DATA_DIR = FIGURES_DIR / "knn" / "data"


def _get_devices():
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return [torch.device(f"cuda:{i}") for i in range(n)] if n > 0 else [torch.device("cpu")]


def run_probes(
    reg_names: list[str],
    cls_names: list[str],
    emb_names: list[str],
    *,
    load_device: str,
    force: bool,
    use_gpu: bool,
    devices: list,
):
    if not devices:
        devices = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]

    probe_knn_regression_fn = probe_knn_regression_gpu if use_gpu else probe_knn_regression
    probe_knn_classification_fn = probe_knn_classification_gpu if use_gpu else probe_knn_classification
    results = []

    for ds_name in reg_names:
        print(f"Dataset (regression): {ds_name} — loading…", flush=True)
        _, by_model, y, _, _ = load_embeddings_and_labels(
            ds_name, embedding_models=emb_names, device=load_device, force=force
        )
        for emb_idx, emb_name in enumerate(emb_names):
            dev = devices[emb_idx % len(devices)]
            X = by_model[emb_name]
            print(f"Probing {emb_name} on {dev} — {ds_name}…", flush=True)
            if use_gpu:
                scores = probe_knn_regression_fn(X, y, device=dev)
            else:
                scores = probe_knn_regression_fn(X, y)
            for k, (mean, std) in scores.items():
                results.append({"embedding": emb_name, "dataset": ds_name,
                                 "task": "regression", "k": k, "score": mean, "std": std})
            summary = ", ".join(f"k={k}: {m:.4f}±{s:.4f}" for k, (m, s) in scores.items())
            print(f"  {ds_name}: {summary}")

    for ds_name in cls_names:
        print(f"Dataset (classification): {ds_name} — loading…", flush=True)
        _, by_model, y, _, _ = load_embeddings_and_labels(
            ds_name, embedding_models=emb_names, device=load_device, force=force
        )
        for emb_idx, emb_name in enumerate(emb_names):
            dev = devices[emb_idx % len(devices)]
            X = by_model[emb_name]
            print(f"Probing {emb_name} on {dev} — {ds_name}…", flush=True)
            if use_gpu:
                scores = probe_knn_classification_fn(X, y, device=dev)
            else:
                scores = probe_knn_classification_fn(X, y)
            for k, (mean, std) in scores.items():
                results.append({"embedding": emb_name, "dataset": ds_name,
                                 "task": "classification", "k": k, "score": mean, "std": std})
            summary = ", ".join(f"k={k}: {m:.4f}±{s:.4f}" for k, (m, s) in scores.items())
            print(f"  {ds_name}: {summary}")

    df = pd.DataFrame(results)

    # Mark best k per (embedding, dataset, task)
    best_k_idx = df.groupby(["embedding", "dataset", "task"])["score"].idxmax()
    df["is_best_k"] = False
    df.loc[best_k_idx, "is_best_k"] = True

    return df


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


def plot_heatmap_per_k(df, task, dataset_names, embedding_names, metric_label, out_dir):
    """Save one heatmap per k value."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in K_VALUES:
        pivot = (
            df[(df["task"] == task) & (df["k"] == k)]
            .pivot(index="dataset", columns="embedding", values="score")
            .reindex(index=dataset_names, columns=embedding_names)
        )
        fig, ax = plt.subplots(figsize=(7, max(3, len(dataset_names))))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
                    cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric_label})
        ax.set_title(f"kNN k={k} — {metric_label}", fontsize=13)
        ax.set_xlabel("Embedding")
        ax.set_ylabel("Dataset")
        plt.tight_layout()
        path = out_dir / f"knn_k{k}_{task}_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_best_k_heatmap(df, task, dataset_names, embedding_names, metric_label, out_path):
    """Heatmap using only each dataset's best k per embedding."""
    best = df[(df["task"] == task) & df["is_best_k"]]
    pivot_score = best.pivot(index="dataset", columns="embedding", values="score").reindex(
        index=dataset_names, columns=embedding_names)
    pivot_k = best.pivot(index="dataset", columns="embedding", values="k").reindex(
        index=dataset_names, columns=embedding_names)
    annot = pivot_score.round(2).astype(str) + "\n(k=" + pivot_k.astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7, max(3, len(dataset_names))))
    sns.heatmap(pivot_score, ax=ax, annot=annot, fmt="", vmin=0, vmax=1,
                cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric_label})
    ax.set_title(f"kNN Best-k {metric_label}", fontsize=13)
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
            best_k = grp.loc[grp["score"].idxmax(), "k"]
            ax.plot(grp["k"], grp["score"], marker="o", label=emb_name)
            ax.axvline(best_k, linestyle="--", alpha=0.3)
        ax.set_title(ds_name, fontsize=11)
        ax.set_xlabel("k")
        ax.set_ylabel(metric_label)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle(f"kNN {task.title()} — Score vs. k  (dashed = best k per embedding)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    devices = _get_devices()
    load_device = "cuda" if torch.cuda.is_available() else "cpu"
    reg_names = list(REGRESSION_DATASETS.keys())
    cls_names = list(CLASSIFICATION_DATASETS.keys())
    emb_names = list(EMBEDDING_BACKENDS.keys())

    print(f"Devices: {devices}")
    print("Running probes (one dataset load at a time)…")
    df = run_probes(
        reg_names,
        cls_names,
        emb_names,
        load_device=load_device,
        force=False,
        use_gpu=torch.cuda.is_available(),
        devices=devices,
    )

    # Save full results (all k values)
    df.to_csv(DATA_DIR / "knn_probing_results.csv", index=False)
    df.to_csv(f"{FIGURES_DIR}/knn_probing_results.csv", index=False)  # keep legacy path
    print(df.to_string())

    # Save per-k CSVs
    for k in K_VALUES:
        df[df["k"] == k].to_csv(DATA_DIR / f"knn_k{k}_results.csv", index=False)

    # Save best-k summary
    best_df = df[df["is_best_k"]].copy()
    best_df.to_csv(DATA_DIR / "knn_best_k_results.csv", index=False)
    print(f"Saved per-k CSVs and best-k CSV to {DATA_DIR}")

    embedding_names = emb_names

    # Combined multi-panel heatmaps (one panel per k)
    plot_heatmaps(df, "regression", reg_names, embedding_names,
                  "R²", FIGURES_DIR / "knn" / "regression" / "knn_regression_heatmap.png")
    plot_heatmaps(df, "classification", cls_names, embedding_names,
                  "Accuracy", FIGURES_DIR / "knn" / "classification" / "knn_classification_heatmap.png")

    # Per-k individual heatmaps
    plot_heatmap_per_k(df, "regression", reg_names, embedding_names,
                       "R²", FIGURES_DIR / "knn" / "regression" / "per_k")
    plot_heatmap_per_k(df, "classification", cls_names, embedding_names,
                       "Accuracy", FIGURES_DIR / "knn" / "classification" / "per_k")

    # Best-k heatmaps
    plot_best_k_heatmap(df, "regression", reg_names, embedding_names,
                        "R²", FIGURES_DIR / "knn" / "regression" / "knn_regression_best_k_heatmap.png")
    plot_best_k_heatmap(df, "classification", cls_names, embedding_names,
                        "Accuracy", FIGURES_DIR / "knn" / "classification" / "knn_classification_best_k_heatmap.png")

    # k-curve plots (score vs k with best-k dashed lines)
    plot_k_curves(df, "regression", "R²", FIGURES_DIR / "knn" / "regression" / "knn_regression_k_curve.png")
    plot_k_curves(df, "classification", "Accuracy", FIGURES_DIR / "knn" / "classification" / "knn_classification_k_curve.png")

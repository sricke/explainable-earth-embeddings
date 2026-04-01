import argparse
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
from probes import probe_ridge, probe_logistic
from gpu_probes import probe_ridge_gpu, probe_logistic_gpu
from utils import load_embeddings_and_labels, EMBEDDING_BACKENDS, REGRESSION_DATASETS, CLASSIFICATION_DATASETS

FIGURES_DIR = Path("../../figures")
DATA_DIR = FIGURES_DIR / "ridge" / "data"


def _get_devices():
    n = 0
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
    """Load **one dataset at a time**; for each, run all selected embedding models."""
    probe_ridge_fn = probe_ridge_gpu if use_gpu else probe_ridge
    probe_logistic_fn = probe_logistic_gpu if use_gpu else probe_logistic
    results = []
    alpha_records = []

    for ds_name in reg_names:
        print(f"Dataset (regression): {ds_name} — loading…", flush=True)
        _, by_model, y, train_idx, test_idx = load_embeddings_and_labels(
            ds_name, embedding_models=emb_names, device=load_device, force=force
        )
        for e in emb_names:
            print(f"  {e}: X={by_model[e].shape}", flush=True)

        for emb_idx, emb_name in enumerate(emb_names):
            dev = devices[emb_idx % len(devices)]
            X = by_model[emb_name]
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            print(f'Probing {emb_name} on {dev} — {ds_name}…', flush=True)
            if use_gpu:
                score_mean, score_std, alpha_info = probe_ridge_fn(
                    X_tr, y_tr, X_te, y_te, device=dev
                )
            else:
                score_mean, score_std, alpha_info = probe_ridge_fn(X_tr, y_tr, X_te, y_te)
            for alpha, mean_cv in alpha_info["mean_cv_r2"].items():
                alpha_records.append({
                    "embedding": emb_name,
                    "dataset": ds_name,
                    "alpha": alpha,
                    "mean_cv_r2": mean_cv,
                    "std_cv_r2": alpha_info["std_cv_r2"][alpha],
                    "n_selected": alpha_info["selection_counts"][alpha],
                    "is_best": alpha == alpha_info["best_alpha"],
                })

            results.append({
                'embedding': emb_name,
                'dataset': ds_name,
                'task': 'regression',
                'score': score_mean,
                'std': score_std,
            })
            print(
                f'  {ds_name}: test R²={score_mean:.4f} '
                f'(train CV σ for best α={score_std:.4f})'
            )

    for ds_name in cls_names:
        print(f"Dataset (classification): {ds_name} — loading…", flush=True)
        _, by_model, y, _train_idx, _test_idx = load_embeddings_and_labels(
            ds_name, embedding_models=emb_names, device=load_device, force=force
        )
        for e in emb_names:
            print(f"  {e}: X={by_model[e].shape}", flush=True)

        for emb_idx, emb_name in enumerate(emb_names):
            dev = devices[emb_idx % len(devices)]
            X = by_model[emb_name]
            print(f'Probing {emb_name} on {dev} — {ds_name}…', flush=True)
            if use_gpu:
                score_mean, score_std = probe_logistic_fn(X, y, device=dev)
            else:
                score_mean, score_std = probe_logistic_fn(X, y)
            results.append({
                'embedding': emb_name,
                'dataset': ds_name,
                'task': 'classification',
                'score': score_mean,
                'std': score_std,
            })
            print(f'  {ds_name}: Accuracy={score_mean:.4f} ± {score_std:.4f}')

    alpha_df = pd.DataFrame(alpha_records) if alpha_records else None
    return pd.DataFrame(results), alpha_df


def _parse_args():
    p = argparse.ArgumentParser(
        description="Ridge / logistic probing; loads one prediction dataset at a time.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for embedding models when caches miss (default: cuda).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Regenerate embedding caches.",
    )
    p.add_argument(
        "--embedding",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of embedding models (e.g. SatCLIP GeoCLIP). Default: all.",
    )
    p.add_argument(
        "--regression",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of regression datasets. Default: all unless --no-regression.",
    )
    p.add_argument(
        "--classification",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Subset of classification datasets. Default: all unless --no-classification.",
    )
    p.add_argument(
        "--no-regression",
        action="store_true",
        help="Skip all regression tasks.",
    )
    p.add_argument(
        "--no-classification",
        action="store_true",
        help="Skip all classification tasks.",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write CSV/plots.",
    )
    return p.parse_args()


def _resolve_lists(args):
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    if args.embedding is not None:
        unknown = set(args.embedding) - set(emb_keys)
        if unknown:
            raise SystemExit(f"Unknown --embedding {unknown}. Choose from: {emb_keys}")
        emb_names = list(args.embedding)
    else:
        emb_names = emb_keys

    if args.no_regression:
        reg_names = []
    elif args.regression is not None:
        unknown = set(args.regression) - set(REGRESSION_DATASETS)
        if unknown:
            raise SystemExit(f"Unknown --regression {unknown}. Choose from: {list(REGRESSION_DATASETS)}")
        reg_names = list(args.regression)
    else:
        reg_names = list(REGRESSION_DATASETS.keys())

    if args.no_classification:
        cls_names = []
    elif args.classification is not None:
        unknown = set(args.classification) - set(CLASSIFICATION_DATASETS)
        if unknown:
            raise SystemExit(f"Unknown --classification {unknown}. Choose from: {list(CLASSIFICATION_DATASETS)}")
        cls_names = list(args.classification)
    else:
        cls_names = list(CLASSIFICATION_DATASETS.keys())

    return emb_names, reg_names, cls_names


def plot_heatmap(df, task, dataset_names, embedding_names, metric_label, out_path):
    sub = df[df["task"] == task]
    pivot = sub.pivot(index="dataset", columns="embedding", values="score").reindex(
        index=dataset_names, columns=embedding_names
    )
    fig, ax = plt.subplots(figsize=(7, max(3, len(dataset_names))))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", vmin=0, vmax=1,
                cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": metric_label})
    title_task = "Ridge Regression (test)" if task == "regression" else "Logistic Regression"
    ax.set_title(f"{title_task} {metric_label}", fontsize=14)
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


def plot_alpha_gcv_curves(alpha_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for emb_name, grp in alpha_df.groupby("embedding"):
        fig, ax = plt.subplots(figsize=(7, 5))
        for ds_name, sub in grp.groupby("dataset"):
            sub = sub.sort_values("alpha")
            ax.errorbar(sub["alpha"], sub["mean_cv_r2"], yerr=sub["std_cv_r2"],
                        marker="o", label=ds_name, capsize=3)
            best = sub[sub["is_best"]]
            ax.scatter(best["alpha"], best["mean_cv_r2"], s=120, zorder=5, marker="*")
        ax.set_xscale("log")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Mean CV R² (train)")
        ax.set_title(f"{emb_name} — Ridge CV R² vs Alpha\n(★ = best alpha per dataset)")
        ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        path = out_dir / f"ridge_alpha_gcv_{emb_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_alpha_selection_heatmap(alpha_df: pd.DataFrame, out_dir: Path):
    pivot = alpha_df.pivot_table(
        index="dataset", columns=["embedding", "alpha"], values="n_selected", aggfunc="sum"
    )
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.6), max(4, len(pivot))))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".0f", cmap="Blues",
                cbar_kws={"label": "# folds selecting this alpha"})
    ax.set_title("Ridge — CV folds attributed to best alpha (train CV)")
    plt.tight_layout()
    path = out_dir / "ridge_alpha_selection_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    args = _parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)
    if not args.no_save:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    devices = _get_devices()
    print(f"Devices: {devices}")

    emb_names, reg_names, cls_names = _resolve_lists(args)

    if not reg_names and not cls_names:
        raise SystemExit("Nothing to probe. Adjust --regression / --classification / --no-* flags.")

    load_device = args.device
    print(f"Embedding load device: {load_device}\n")

    print("Running probes (one dataset load at a time)…")
    df, alpha_df = run_probes(
        reg_names,
        cls_names,
        emb_names,
        load_device=load_device,
        force=args.force,
        use_gpu=False,
        devices=devices,
    )
    print(df.to_string())

    if not args.no_save:
        df.to_csv(DATA_DIR / "ridge_probing_results.csv", index=False)
        df.to_csv(f"{FIGURES_DIR}/ridge_probing_results.csv", index=False)

        if alpha_df is not None and not alpha_df.empty:
            alpha_df.to_csv(DATA_DIR / "ridge_alpha_cv.csv", index=False)
            print(f"Saved alpha CV info → {DATA_DIR / 'ridge_alpha_cv.csv'}")
            alpha_plot_dir = FIGURES_DIR / "ridge" / "alpha_cv"
            plot_alpha_gcv_curves(alpha_df, alpha_plot_dir)
            plot_alpha_selection_heatmap(alpha_df, alpha_plot_dir)

        if reg_names:
            (FIGURES_DIR / "ridge" / "regression").mkdir(parents=True, exist_ok=True)
            plot_heatmap(df, "regression", reg_names, emb_names,
                         "Test R²", FIGURES_DIR / "ridge" / "regression" / "ridge_regression_heatmap.png")
            plot_bars(df, "regression", "Test R²", FIGURES_DIR / "ridge" / "regression" / "ridge_regression_bars.png")
        if cls_names:
            (FIGURES_DIR / "ridge" / "classification").mkdir(parents=True, exist_ok=True)
            plot_heatmap(df, "classification", cls_names, emb_names,
                         "Accuracy", FIGURES_DIR / "ridge" / "classification" / "ridge_classification_heatmap.png")
            plot_bars(df, "classification", "Accuracy", FIGURES_DIR / "ridge" / "classification" / "ridge_classification_bars.png")

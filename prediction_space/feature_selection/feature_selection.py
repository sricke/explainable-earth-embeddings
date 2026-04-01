import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import torch
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.append("..")
from linear_probing.utils import (
    load_embeddings_and_labels,
    REGRESSION_DATASETS,
    CLASSIFICATION_DATASETS,
    EMBEDDING_BACKENDS,
)
from selection_models import (
    fit_sparse_regression, fit_sparse_classification,
    stability_selection, compute_mi, mrmr, progressive_ablation,
)
from selection_models_gpu import (
    fit_sparse_regression_gpu, fit_sparse_classification_gpu,
    stability_selection_gpu, compute_mi_gpu, mrmr_gpu, progressive_ablation_gpu,
)
from utils import (
    plot_importance_heatmap, plot_sparsity_barchart,
    plot_stability_heatmap, plot_mi_heatmap,
    plot_mrmr_heatmap, plot_ablation_curves,
)

FIGURES_DIR = Path("../figures")
DATA_DIR    = FIGURES_DIR / "feature_selection" / "data"
N_MRMR = 32

STABILITY_TASKS = REGRESSION_DATASETS | CLASSIFICATION_DATASETS
ABLATION_TASKS  = REGRESSION_DATASETS | CLASSIFICATION_DATASETS


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_devices(n_gpus: int) -> list[str]:
    if torch.cuda.is_available() and n_gpus > 0:
        return [f"cuda:{i}" for i in range(n_gpus)]
    return ["cpu"]


def _assign_device(idx: int, devices: list[str]) -> str:
    return devices[idx % len(devices)]


# ---------------------------------------------------------------------------
# Alpha-CV plotting helpers
# ---------------------------------------------------------------------------

def plot_alpha_cv_curves(alpha_cv_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for emb_name, grp in alpha_cv_df.groupby("embedding"):
        r2_cols = [c for c in grp.columns if c.startswith("alpha_") and c.endswith("_r2")]
        if not r2_cols:
            continue
        alphas = [float(c.replace("alpha_", "").replace("_r2", "")) for c in r2_cols]
        for method in ["lasso", "enet"]:
            sub = grp[grp["method"] == method]
            fig, ax = plt.subplots(figsize=(8, 5))
            for _, row in sub.iterrows():
                r2s = [row[c] for c in r2_cols]
                best_a = row["best_alpha"]
                ax.plot(alphas, r2s, marker="o", label=row["dataset"])
                ax.axvline(best_a, color="gray", linestyle="--", alpha=0.3)
            ax.set_xscale("log")
            ax.set_xlabel("Alpha (regularisation strength)")
            ax.set_ylabel("CV R²")
            ax.set_title(f"{emb_name} — {method.upper()} CV R² vs Alpha")
            ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            path = out_dir / f"alpha_cv_{emb_name}_{method}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved {path}")


def plot_best_alpha_heatmap(alpha_cv_df: pd.DataFrame, out_dir: Path):
    import seaborn as sns
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in ["lasso", "enet"]:
        sub = alpha_cv_df[alpha_cv_df["method"] == method]
        pivot = sub.pivot_table(index="dataset", columns="embedding", values="best_alpha", aggfunc="first")
        fig, ax = plt.subplots(figsize=(7, max(3, len(pivot))))
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3g", cmap="YlOrRd",
                    cbar_kws={"label": "Best alpha"})
        ax.set_title(f"{method.upper()} — Best CV Alpha per Dataset × Embedding")
        plt.tight_layout()
        path = out_dir / f"best_alpha_heatmap_{method}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Analysis functions — each loads from cache or computes
# ---------------------------------------------------------------------------

def run_sparse(
    task_names: list,
    task_types: dict,
    devices,
    use_gpu,
    force=False,
    load_device: str = "cuda",
):
    """Return (importance_matrices, alpha_cv_df). Loads one dataset at a time when computing."""
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    all_cached = all(
        (DATA_DIR / f"importance_matrices_{emb_name}.npz").exists()
        for emb_name in emb_keys
    )
    if all_cached and not force:
        print("Loading sparse model results from cache...")
        importance_matrices = {}
        for emb_name in emb_keys:
            d = np.load(DATA_DIR / f"importance_matrices_{emb_name}.npz", allow_pickle=True)
            importance_matrices[emb_name] = {"lasso": d["lasso"], "enet": d["enet"]}
        alpha_cv_path = DATA_DIR / "sparse_regression_alpha_cv.csv"
        alpha_cv_df = pd.read_csv(alpha_cv_path) if alpha_cv_path.exists() else None
        return importance_matrices, alpha_cv_df

    print("Fitting sparse models (LASSO / Elastic Net)...")

    sparse_tasks = []
    for ds_name in task_names:
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        tt = task_types[ds_name]
        for emb_name in emb_keys:
            sparse_tasks.append((emb_name, ds_name, by_model[emb_name], y, tt))

    def _run_sparse(args):
        emb_name, ds_name, X, y, task_type, device = args
        if task_type == "regression":
            lasso_imp, enet_imp, cv_info = fit_sparse_regression_gpu(X, y, device=device)
        else:
            lasso_imp, enet_imp = fit_sparse_classification_gpu(X, y, device=device)
            cv_info = None
        return emb_name, ds_name, task_type, lasso_imp, enet_imp, cv_info

    def _run_sparse_cpu(args):
        emb_name, ds_name, X, y, task_type = args
        if task_type == "regression":
            lasso_imp, enet_imp = fit_sparse_regression(X, y)
        else:
            lasso_imp, enet_imp = fit_sparse_classification(X, y)
        return emb_name, ds_name, task_type, lasso_imp, enet_imp, None

    sparse_results = {}
    if use_gpu:
        gpu_tasks = [
            (emb_name, ds_name, X, y, task_type, _assign_device(i, devices))
            for i, (emb_name, ds_name, X, y, task_type) in enumerate(sparse_tasks)
        ]
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(_run_sparse, t): (t[0], t[1]) for t in gpu_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Sparse (GPU)"):
                emb_name, ds_name, task_type, lasso_imp, enet_imp, cv_info = fut.result()
                sparse_results[(emb_name, ds_name)] = (lasso_imp, enet_imp, cv_info)
    else:
        for t in tqdm(sparse_tasks, desc="Sparse (CPU)"):
            emb_name, ds_name, task_type, lasso_imp, enet_imp, cv_info = _run_sparse_cpu(t)
            sparse_results[(emb_name, ds_name)] = (lasso_imp, enet_imp, cv_info)

    coef_store = defaultdict(lambda: {"lasso": {}, "enet": {}})
    importance_matrices = defaultdict(lambda: {"lasso": [], "enet": []})
    alpha_cv_records = []

    def normalize(v):
        v = np.asarray(v)
        return v / (np.max(v) + 1e-12)

    for emb_name in emb_keys:
        for ds_name in task_names:
            lasso_imp, enet_imp, cv_info = sparse_results[(emb_name, ds_name)]
            coef_store[emb_name]["lasso"][ds_name] = lasso_imp
            coef_store[emb_name]["enet"][ds_name]  = enet_imp

            lasso_vec = np.abs(lasso_imp).max(axis=0) if lasso_imp.ndim == 2 else np.abs(lasso_imp)
            enet_vec  = np.abs(enet_imp).max(axis=0)  if enet_imp.ndim == 2  else np.abs(enet_imp)

            task_type = task_types[ds_name]
            print(
                f"  [{emb_name}] {ds_name} ({task_type}): "
                f"lasso max={lasso_vec.max():.4e} nnz={(lasso_vec > 1e-10).sum()}/{len(lasso_vec)}, "
                f"enet  max={enet_vec.max():.4e}  nnz={(enet_vec  > 1e-10).sum()}/{len(enet_vec)}"
            )

            importance_matrices[emb_name]["lasso"].append(normalize(lasso_vec))
            importance_matrices[emb_name]["enet"].append(normalize(enet_vec))

            if cv_info is not None:
                for method in ["lasso", "enet"]:
                    rec = {
                        "embedding":  emb_name,
                        "dataset":    ds_name,
                        "method":     method,
                        "best_alpha": cv_info[method]["best_alpha"],
                        "cv_r2":      cv_info[method]["cv_r2"],
                    }
                    for alpha, r2 in cv_info[method]["alpha_r2"].items():
                        rec[f"alpha_{alpha}_r2"] = r2
                    for alpha, mse in cv_info[method]["alpha_cv_mse"].items():
                        rec[f"alpha_{alpha}_mse"] = mse
                    alpha_cv_records.append(rec)

        importance_matrices[emb_name]["lasso"] = np.stack(importance_matrices[emb_name]["lasso"], axis=1)
        importance_matrices[emb_name]["enet"]  = np.stack(importance_matrices[emb_name]["enet"],  axis=1)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for emb_name, methods in coef_store.items():
        np.savez(
            DATA_DIR / f"coef_store_{emb_name}.npz",
            task_names=np.array(task_names),
            **{f"lasso__{ds}": methods["lasso"][ds] for ds in methods["lasso"]},
            **{f"enet__{ds}":  methods["enet"][ds]  for ds in methods["enet"]},
        )
    for emb_name, mats in importance_matrices.items():
        np.savez(
            DATA_DIR / f"importance_matrices_{emb_name}.npz",
            lasso=mats["lasso"],
            enet=mats["enet"],
            task_names=np.array(task_names),
        )
        print(f"Saved importance matrices for {emb_name}")

    alpha_cv_df = None
    if alpha_cv_records:
        alpha_cv_df = pd.DataFrame(alpha_cv_records)
        alpha_cv_path = DATA_DIR / "sparse_regression_alpha_cv.csv"
        alpha_cv_df.to_csv(alpha_cv_path, index=False)
        print(f"Saved {alpha_cv_path}")

    return dict(importance_matrices), alpha_cv_df


def run_ablation(
    task_names: list,
    task_types: dict,
    importance_matrices,
    devices,
    use_gpu,
    force=False,
    load_device: str = "cuda",
):
    """Return (ablation_curves, ablation_task_types). Loads one dataset at a time when computing."""
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    ablation_task_labels = {
        name: task_types[name] for name in ABLATION_TASKS if name in task_types
    }
    ablation_dir = DATA_DIR / "ablation"

    all_cached = all(
        (ablation_dir / f"ablation_{ds_name}_{emb_name}.npz").exists()
        for emb_name in emb_keys
        for ds_name in ablation_task_labels
    )
    if all_cached and not force:
        print("Loading ablation curves from cache...")
        ablation_curves = {ds: {} for ds in ablation_task_labels}
        for emb_name in emb_keys:
            for ds_name in ablation_task_labels:
                d = np.load(ablation_dir / f"ablation_{ds_name}_{emb_name}.npz")
                ablation_curves[ds_name][emb_name] = (d["fracs"], d["scores"])
        ablation_task_types = {name: ablation_task_labels[name] for name in ablation_task_labels}
        return ablation_curves, ablation_task_types

    print("Running progressive ablation...")
    ablation_task_types = dict(ablation_task_labels)
    ablation_curves = {ds: {} for ds in ablation_task_labels}

    def _run_ablation(args):
        emb_name, ds_name, X, y, task_type, imp, device = args
        fracs, scores = progressive_ablation_gpu(X, y, task_type, imp, device=device)
        return emb_name, ds_name, fracs, scores

    ablation_list = list(ablation_task_labels.items())
    ablation_tasks = []
    for j, (ds_name, task_type) in enumerate(ablation_list):
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        for i, emb_name in enumerate(emb_keys):
            X = by_model[emb_name]
            imp = importance_matrices[emb_name]["lasso"][:, task_names.index(ds_name)]
            ablation_tasks.append((
                emb_name, ds_name, X, y, task_type, imp,
                _assign_device(i * len(ablation_list) + j, devices),
            ))

    if use_gpu:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(_run_ablation, t): (t[0], t[1]) for t in ablation_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Ablation (GPU)"):
                emb_name, ds_name, fracs, scores = fut.result()
                ablation_curves[ds_name][emb_name] = (fracs, scores)
    else:
        for t in tqdm(ablation_tasks, desc="Ablation (CPU)"):
            emb_name, ds_name, X, y, task_type, imp, _ = t
            fracs, scores = progressive_ablation(X, y, task_type, imp)
            ablation_curves[ds_name][emb_name] = (fracs, scores)

    ablation_dir.mkdir(parents=True, exist_ok=True)
    for ds_name, emb_dict in ablation_curves.items():
        for emb_name, (fracs, scores) in emb_dict.items():
            np.savez(
                ablation_dir / f"ablation_{ds_name}_{emb_name}.npz",
                fracs=np.array(fracs),
                scores=np.array(scores),
            )
    print(f"Saved ablation curves to {ablation_dir}")

    return ablation_curves, ablation_task_types


def run_stability(
    task_names: list,
    task_types: dict,
    devices,
    use_gpu,
    force=False,
    load_device: str = "cuda",
):
    """Return (stability_freqs, stability_task_names). Loads one dataset at a time when computing."""
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    stability_task_labels = {
        name: task_types[name] for name in STABILITY_TASKS if name in task_types
    }

    all_cached = all(
        (DATA_DIR / f"stability_freqs_{emb_name}.npz").exists()
        for emb_name in emb_keys
    )
    if all_cached and not force:
        print("Loading stability selection results from cache...")
        stability_freqs = {}
        for emb_name in emb_keys:
            d = np.load(DATA_DIR / f"stability_freqs_{emb_name}.npz", allow_pickle=True)
            stability_freqs[emb_name] = d["freqs"]
        return stability_freqs, list(stability_task_labels.keys())

    print("Running stability selection...")

    def _run_stability(args):
        emb_name, ds_name, X, y, task_type, device = args
        freq = stability_selection_gpu(X, y, task_type, device=device)
        return emb_name, ds_name, freq

    stab_list = list(stability_task_labels.items())
    stability_tasks = []
    for j, (ds_name, task_type) in enumerate(stab_list):
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        for i, emb_name in enumerate(emb_keys):
            stability_tasks.append((
                emb_name, ds_name, by_model[emb_name], y, task_type,
                _assign_device(i * len(stab_list) + j, devices),
            ))

    stab_results = {}
    if use_gpu:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(_run_stability, t): (t[0], t[1]) for t in stability_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Stability (GPU)"):
                emb_name, ds_name, freq = fut.result()
                stab_results[(emb_name, ds_name)] = freq
    else:
        for t in tqdm(stability_tasks, desc="Stability (CPU)"):
            emb_name, ds_name, X, y, task_type, _ = t
            stab_results[(emb_name, ds_name)] = stability_selection(X, y, task_type)

    stability_freqs = {}
    for emb_name in emb_keys:
        cols = [stab_results[(emb_name, ds_name)] for ds_name in stability_task_labels]
        stability_freqs[emb_name] = np.stack(cols, axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for emb_name, mat in stability_freqs.items():
        np.savez(
            DATA_DIR / f"stability_freqs_{emb_name}.npz",
            freqs=mat,
            task_names=np.array(list(stability_task_labels.keys())),
        )
        print(f"Saved stability_freqs for {emb_name}")

    return stability_freqs, list(stability_task_labels.keys())


def run_mi(
    task_names: list,
    task_types: dict,
    devices,
    use_gpu,
    force=False,
    load_device: str = "cuda",
):
    """Return mi_matrices. Loads one dataset at a time when computing."""
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    all_cached = all(
        (DATA_DIR / f"mi_matrix_{emb_name}.npz").exists()
        for emb_name in emb_keys
    )
    if all_cached and not force:
        print("Loading MI matrices from cache...")
        mi_matrices = {}
        for emb_name in emb_keys:
            d = np.load(DATA_DIR / f"mi_matrix_{emb_name}.npz", allow_pickle=True)
            mi_matrices[emb_name] = d["mi"]
        return mi_matrices

    print("Computing mutual information...")

    def _run_mi(args):
        emb_name, ds_name, X, y, task_type, device = args
        mi = compute_mi_gpu(X, y, task_type, device=device)
        return emb_name, ds_name, mi

    mi_tasks = []
    for j, ds_name in enumerate(task_names):
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        tt = task_types[ds_name]
        for i, emb_name in enumerate(emb_keys):
            mi_tasks.append((
                emb_name, ds_name, by_model[emb_name], y, tt,
                _assign_device(i * len(task_names) + j, devices),
            ))

    mi_results = {}
    if use_gpu:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(_run_mi, t): (t[0], t[1]) for t in mi_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="MI (GPU)"):
                emb_name, ds_name, mi = fut.result()
                mi_results[(emb_name, ds_name)] = mi
    else:
        for t in tqdm(mi_tasks, desc="MI (CPU)"):
            emb_name, ds_name, X, y, task_type, _ = t
            mi_results[(emb_name, ds_name)] = compute_mi(X, y, task_type)

    mi_matrices = {}
    for emb_name in emb_keys:
        cols = [mi_results[(emb_name, ds_name)] for ds_name in task_names]
        mi_matrices[emb_name] = np.stack(cols, axis=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for emb_name, mat in mi_matrices.items():
        np.savez(
            DATA_DIR / f"mi_matrix_{emb_name}.npz",
            mi=mat,
            task_names=np.array(task_names),
        )
        print(f"Saved MI matrix for {emb_name}")

    return mi_matrices


def run_mrmr(
    task_names: list,
    task_types: dict,
    devices,
    use_gpu,
    force=False,
    load_device: str = "cuda",
):
    """Return mrmr_results. Loads one dataset at a time when computing."""
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    mrmr_path = DATA_DIR / "mrmr_results.json"
    if mrmr_path.exists() and not force:
        print("Loading mRMR results from cache...")
        with open(mrmr_path) as f:
            mrmr_results = json.load(f)
        # JSON keys are strings; convert dim indices back to int
        mrmr_results = {
            emb: {ds: [int(x) for x in idxs] for ds, idxs in task_dict.items()}
            for emb, task_dict in mrmr_results.items()
        }
        return mrmr_results

    print("Running mRMR...")
    mrmr_results = {name: {} for name in emb_keys}

    def _run_mrmr(args):
        emb_name, ds_name, X, y, task_type, device = args
        selected = mrmr_gpu(X, y, task_type, n_features=N_MRMR, device=device)
        return emb_name, ds_name, selected

    mrmr_tasks = []
    for j, ds_name in enumerate(task_names):
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        tt = task_types[ds_name]
        for i, emb_name in enumerate(emb_keys):
            mrmr_tasks.append((
                emb_name, ds_name, by_model[emb_name], y, tt,
                _assign_device(i * len(task_names) + j, devices),
            ))

    if use_gpu:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(_run_mrmr, t): (t[0], t[1]) for t in mrmr_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="mRMR (GPU)"):
                emb_name, ds_name, selected = fut.result()
                mrmr_results[emb_name][ds_name] = selected
    else:
        for t in tqdm(mrmr_tasks, desc="mRMR (CPU)"):
            emb_name, ds_name, X, y, task_type, _ = t
            mrmr_results[emb_name][ds_name] = mrmr(X, y, task_type, n_features=N_MRMR)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(mrmr_path, "w") as f:
        json.dump(mrmr_results, f)
    print(f"Saved {mrmr_path}")

    return mrmr_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature selection interpretability for earth embeddings.")
    parser.add_argument("--lasso",     action="store_true", help="Fit LASSO and plot importance heatmap")
    parser.add_argument("--enet",      action="store_true", help="Fit Elastic Net and plot importance heatmap")
    parser.add_argument("--ablation",  action="store_true", help="Run progressive ablation")
    parser.add_argument("--stability", action="store_true", help="Run stability selection")
    parser.add_argument("--mi",        action="store_true", help="Compute mutual information")
    parser.add_argument("--mrmr",      action="store_true", help="Run mRMR feature ranking")
    parser.add_argument("--all",       action="store_true", help="Run all analyses")
    parser.add_argument("--rerun",     action="store_true", help="Recompute even if cached results exist")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_gpus  = torch.cuda.device_count() if torch.cuda.is_available() else 0
    devices = _get_devices(n_gpus)
    use_gpu = torch.cuda.is_available()
    load_device = devices[0] if devices else "cpu"
    print(f"GPUs available: {n_gpus}  →  devices: {devices}")

    task_names = (
        list(REGRESSION_DATASETS.keys()) + list(CLASSIFICATION_DATASETS.keys())
    )
    task_types = {
        **{k: "regression" for k in REGRESSION_DATASETS},
        **{k: "classification" for k in CLASSIFICATION_DATASETS},
    }
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    print(f"Tasks: {len(task_names)} datasets (lazy load per analysis); embeddings: {emb_keys}")

    # Sparse models are needed for lasso, enet, and ablation plots
    need_sparse = args.lasso or args.enet or args.ablation or args.all
    importance_matrices = None
    if need_sparse:
        importance_matrices, alpha_cv_df = run_sparse(
            task_names,
            task_types,
            devices,
            use_gpu,
            force=args.rerun,
            load_device=load_device,
        )
        if alpha_cv_df is not None:
            alpha_plot_dir = FIGURES_DIR / "feature_selection" / "lasso" / "alpha_cv"
            plot_alpha_cv_curves(alpha_cv_df, alpha_plot_dir)
            plot_best_alpha_heatmap(alpha_cv_df, alpha_plot_dir)

    if args.lasso or args.all:
        plot_importance_heatmap(
            importance_matrices, task_names, method="lasso",
            out_path=FIGURES_DIR / "feature_selection" / "lasso" / "2a_lasso_importance_heatmap.png",
        )
        plot_sparsity_barchart(
            importance_matrices, task_names,
            out_path=FIGURES_DIR / "feature_selection" / "2b_sparsity_barchart.png",
        )

    if args.enet or args.all:
        plot_importance_heatmap(
            importance_matrices, task_names, method="enet",
            out_path=FIGURES_DIR / "feature_selection" / "enet" / "2a_enet_importance_heatmap.png",
        )

    if args.ablation or args.all:
        ablation_curves, ablation_task_types = run_ablation(
            task_names,
            task_types,
            importance_matrices,
            devices,
            use_gpu,
            force=args.rerun,
            load_device=load_device,
        )
        plot_ablation_curves(
            ablation_curves, ablation_task_types, emb_keys,
            out_path=FIGURES_DIR / "feature_selection" / "progressive_ablation" / "2f_progressive_ablation.png",
        )

    if args.stability or args.all:
        stability_freqs, stability_task_names = run_stability(
            task_names,
            task_types,
            devices,
            use_gpu,
            force=args.rerun,
            load_device=load_device,
        )
        print("Stability frequencies (sample):")
        for emb_name, freqs in stability_freqs.items():
            print(f"  {emb_name}: shape={freqs.shape}, sample={freqs[:5, :5]}")
        plot_stability_heatmap(
            stability_freqs, stability_task_names,
            out_path=FIGURES_DIR / "feature_selection" / "stability_selection" / "2c_stability_selection_heatmap.png",
        )

    if args.mi or args.all:
        mi_matrices = run_mi(
            task_names,
            task_types,
            devices,
            use_gpu,
            force=args.rerun,
            load_device=load_device,
        )
        print("Mutual information matrices (sample):")
        for emb_name, mi_mat in mi_matrices.items():
            print(f"  {emb_name}: shape={mi_mat.shape}, sample={mi_mat[:5, :5]}")
        plot_mi_heatmap(
            mi_matrices, task_names,
            out_path=FIGURES_DIR / "feature_selection" / "mutual_information" / "2d_mi_heatmap.png",
        )

    if args.mrmr or args.all:
        mrmr_results = run_mrmr(
            task_names,
            task_types,
            devices,
            use_gpu,
            force=args.rerun,
            load_device=load_device,
        )
        plot_mrmr_heatmap(
            mrmr_results, task_names, n_mrmr=N_MRMR,
            out_path=FIGURES_DIR / "feature_selection" / "mrmr" / "2e_mrmr_rank_heatmap.png",
        )

    print("\nDone.")

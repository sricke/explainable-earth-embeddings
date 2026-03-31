import sys
import warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

sys.path.append("..")
from linear_probing.utils import load_embeddings, load_labels, REGRESSION_DATASETS, CLASSIFICATION_DATASETS
from selection_models import (
    fit_sparse_regression, fit_sparse_classification,
    stability_selection, compute_mi, mrmr, progressive_ablation,
)
from utils import (
    plot_importance_heatmap, plot_sparsity_barchart,
    plot_stability_heatmap, plot_mi_heatmap,
    plot_mrmr_heatmap, plot_ablation_curves,
)

FIGURES_DIR = Path("../figures")
N_MRMR = 32
TOP_K = 64

STABILITY_TASKS = ["Elevation", "Temperature", "Biome", "Climate Zone"]
ABLATION_TASKS  = ["Elevation", "Temperature", "Biome", "Climate Zone"]


if __name__ == "__main__":
    FIGURES_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading embeddings and labels...")
    latlons, embeddings = load_embeddings()
    regression_labels, classification_labels = load_labels(latlons)

    all_labels = {
        **{k: (v, "regression")     for k, v in regression_labels.items()},
        **{k: (v, "classification") for k, v in classification_labels.items()},
    }
    task_names = list(all_labels.keys())

    for name, emb in embeddings.items():
        print(f"  {name}: shape={emb.shape}")

    # ------------------------------------------------------------------
    # LASSO & Elastic Net importance
    # ------------------------------------------------------------------
    print("\nFitting sparse models (LASSO / Elastic Net)...")
    importance_matrices = {name: {"lasso": [], "enet": []} for name in embeddings}

    for emb_name, X in embeddings.items():
        print(f"  {emb_name}")
        for ds_name, (y, task_type) in tqdm(all_labels.items(), desc="  Datasets"):
            if task_type == "regression":
                lasso_imp, enet_imp = fit_sparse_regression(X, y)
            else:
                lasso_imp, enet_imp = fit_sparse_classification(X, y)
            importance_matrices[emb_name]["lasso"].append(lasso_imp)
            importance_matrices[emb_name]["enet"].append(enet_imp)

        importance_matrices[emb_name]["lasso"] = np.stack(importance_matrices[emb_name]["lasso"], axis=1)
        importance_matrices[emb_name]["enet"]  = np.stack(importance_matrices[emb_name]["enet"],  axis=1)
        print(f"    importance shape: {importance_matrices[emb_name]['lasso'].shape}")

    plot_importance_heatmap(importance_matrices, embeddings, task_names, method="lasso",
                             top_k=TOP_K, out_path=FIGURES_DIR / "2a_lasso_importance_heatmap.png")
    plot_sparsity_barchart(importance_matrices, task_names,
                            out_path=FIGURES_DIR / "2b_sparsity_barchart.png")

    # ------------------------------------------------------------------
    # Stability selection
    # ------------------------------------------------------------------
    print("\nRunning stability selection...")
    stability_freqs = {}
    stability_task_labels = {
        name: all_labels[name] for name in STABILITY_TASKS if name in all_labels
    }

    for emb_name, X in embeddings.items():
        cols = []
        for ds_name, (y, task_type) in tqdm(stability_task_labels.items(), desc=f"  {emb_name}"):
            cols.append(stability_selection(X, y, task_type))
        stability_freqs[emb_name] = np.stack(cols, axis=1)

    plot_stability_heatmap(stability_freqs, list(stability_task_labels.keys()),
                            top_k=TOP_K, out_path=FIGURES_DIR / "2c_stability_selection_heatmap.png")

    # ------------------------------------------------------------------
    # Mutual information
    # ------------------------------------------------------------------
    print("\nComputing mutual information...")
    mi_matrices = {}

    for emb_name, X in embeddings.items():
        cols = []
        for ds_name, (y, task_type) in tqdm(all_labels.items(), desc=f"  {emb_name}"):
            cols.append(compute_mi(X, y, task_type))
        mi_matrices[emb_name] = np.stack(cols, axis=1)

    plot_mi_heatmap(mi_matrices, task_names,
                    top_k=TOP_K, out_path=FIGURES_DIR / "2d_mi_heatmap.png")

    # ------------------------------------------------------------------
    # mRMR
    # ------------------------------------------------------------------
    print("\nRunning mRMR...")
    mrmr_results = {name: {} for name in embeddings}

    for emb_name, X in embeddings.items():
        for ds_name, (y, task_type) in tqdm(all_labels.items(), desc=f"  {emb_name}"):
            mrmr_results[emb_name][ds_name] = mrmr(X, y, task_type, n_features=N_MRMR)

    plot_mrmr_heatmap(mrmr_results, embeddings, task_names,
                       n_mrmr=N_MRMR, out_path=FIGURES_DIR / "2e_mrmr_rank_heatmap.png")

    # ------------------------------------------------------------------
    # Progressive ablation
    # ------------------------------------------------------------------
    print("\nRunning progressive ablation...")
    ablation_task_labels = {
        name: all_labels[name] for name in ABLATION_TASKS if name in all_labels
    }
    ablation_task_types = {name: task_type for name, (_, task_type) in ablation_task_labels.items()}
    ablation_curves = {ds: {} for ds in ablation_task_labels}

    for emb_name, X in embeddings.items():
        for ds_name, (y, task_type) in tqdm(ablation_task_labels.items(), desc=f"  {emb_name}"):
            global_idx = task_names.index(ds_name)
            imp = importance_matrices[emb_name]["lasso"][:, global_idx]
            fracs, scores = progressive_ablation(X, y, task_type, imp)
            ablation_curves[ds_name][emb_name] = (fracs, scores)

    plot_ablation_curves(ablation_curves, ablation_task_types, embeddings,
                          out_path=FIGURES_DIR / "2f_progressive_ablation.png")

    print("\nDone.")

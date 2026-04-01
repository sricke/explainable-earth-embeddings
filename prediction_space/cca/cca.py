"""
CCA interpretability analysis for earth embeddings.

Two complementary views:

  --cca      Variable-centric: for each environmental variable, which embedding
             dimensions are most aligned with it? (CCA loadings per task)

  --dim-corr Dimension-centric: for each embedding dimension, which environmental
             variables does it correlate with? (Pearson r matrix, dims × labels)
"""

import sys
import json
import argparse
import warnings
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

sys.path.append("..")
from linear_probing.utils import (
    load_embeddings_and_labels,
    REGRESSION_DATASETS,
    CLASSIFICATION_DATASETS,
    EMBEDDING_BACKENDS,
)
from cca_models import compute_cca, compute_dim_label_correlations
from utils import (
    plot_cca_weight_heatmap,
    plot_variable_loadings,
    plot_dim_label_correlations,
    plot_canonical_correlations,
    plot_vars_to_dim_weights,
)

FIGURES_DIR  = Path("../figures")
DATA_DIR     = FIGURES_DIR / "cca" / "data"
N_COMPONENTS = 3


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def run_cca(task_names: list, task_types: dict, load_device: str, force=False):
    """
    Per-task CCA: fit CCA(X=embedding, Y=task_label) for every (embedding, task) pair.
    Loads **one dataset at a time** (all embedding models for that dataset per load).
    """
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    all_cached = all(
        (DATA_DIR / f"cca_{emb_name}.npz").exists()
        for emb_name in emb_keys
    )
    # Check that cache contains weights and loadings
    if all_cached and not force:
        try:
            d = np.load(DATA_DIR / f"cca_{emb_keys[0]}.npz", allow_pickle=True)
            if f"loadings__{task_names[0]}" not in d or f"weights__{task_names[0]}" not in d:
                raise KeyError("Old cache format — missing loadings or weights")
        except (KeyError, IndexError):
            all_cached = False

    if all_cached:
        print("Loading CCA results from cache...")
        cca_weights  = {name: {} for name in emb_keys}
        cca_loadings = {name: {} for name in emb_keys}
        cca_corrs    = {name: {} for name in emb_keys}
        for emb_name in emb_keys:
            d = np.load(DATA_DIR / f"cca_{emb_name}.npz", allow_pickle=True)
            for ds_name in task_names:
                cca_weights[emb_name][ds_name]  = d[f"weights__{ds_name}"]
                cca_loadings[emb_name][ds_name] = d[f"loadings__{ds_name}"]
                cca_corrs[emb_name][ds_name]    = d[f"corrs__{ds_name}"]
        return cca_weights, cca_loadings, cca_corrs

    print(f"Computing CCA (n_components={N_COMPONENTS})...")
    cca_weights  = {name: {} for name in emb_keys}
    cca_loadings = {name: {} for name in emb_keys}
    cca_corrs    = {name: {} for name in emb_keys}

    cca_tasks = []
    for ds_name in task_names:
        _, by_model, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        tt = task_types[ds_name]
        for emb_name in emb_keys:
            cca_tasks.append((emb_name, ds_name, by_model[emb_name], y, tt))

    def _run_cca(args):
        emb_name, ds_name, X, y, task_type = args
        weights, loadings, corrs, n_comp = compute_cca(X, y, task_type, n_components=N_COMPONENTS)
        return emb_name, ds_name, task_type, weights, loadings, corrs, n_comp

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_cca, t): (t[0], t[1]) for t in cca_tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CCA"):
            emb_name, ds_name, task_type, weights, loadings, corrs, n_comp = fut.result()
            cca_weights[emb_name][ds_name]  = weights
            cca_loadings[emb_name][ds_name] = loadings
            cca_corrs[emb_name][ds_name]    = corrs
            comp_strs = [f"comp{i+1}: r={corrs[i]:.3f}" for i in range(n_comp)]
            note = (f"  ← only {n_comp}/{N_COMPONENTS} (Y is 1-D for {task_type})"
                    if n_comp < N_COMPONENTS else "")
            print(f"  [{emb_name}] {ds_name}: {', '.join(comp_strs)}{note}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for emb_name in emb_keys:
        arrays = {}
        for ds_name in task_names:
            arrays[f"weights__{ds_name}"]  = cca_weights[emb_name][ds_name]
            arrays[f"loadings__{ds_name}"] = cca_loadings[emb_name][ds_name]
            arrays[f"corrs__{ds_name}"]    = cca_corrs[emb_name][ds_name]
        np.savez(DATA_DIR / f"cca_{emb_name}.npz", task_names=np.array(task_names), **arrays)
        print(f"Saved CCA results for {emb_name}")

    corrs_summary = {
        emb: {ds: corrs.tolist() for ds, corrs in task_dict.items()}
        for emb, task_dict in cca_corrs.items()
    }
    with open(DATA_DIR / "cca_correlations.json", "w") as f:
        json.dump(corrs_summary, f, indent=2)
    print(f"Saved {DATA_DIR / 'cca_correlations.json'}")

    return cca_weights, cca_loadings, cca_corrs


def run_vars_to_dim_cca(task_names: list, task_types: dict, force=False):
    """
    Dim-centric CCA: for each embedding dimension, find the linear combination of
    environmental variables most correlated with it.

    Runs CCA(X=all_regression_env_vars, Y=embedding_dim) for each dim.
    Returns:
      vars_to_dim_weights[emb_name] : (n_dims, n_reg_vars) — projection weights on env-vars side
      vars_to_dim_corrs[emb_name]   : (n_dims,)            — canonical correlation per dim
      reg_task_names                : ordered list of env-var names used as X
    """
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    reg_task_names = [k for k in task_names if task_types[k] == "regression"]

    cache_path = DATA_DIR / "vars_to_dim_cca.npz"
    if cache_path.exists() and not force:
        print("Loading vars-to-dim CCA results from cache...")
        d = np.load(cache_path, allow_pickle=True)
        weights = {emb: d[f"weights_{emb}"] for emb in emb_keys}
        corrs   = {emb: d[f"corrs_{emb}"]   for emb in emb_keys}
        return weights, corrs, reg_task_names

    # Needs one aligned sample index for all regression targets and the embedding matrix.
    # Per-task ``load_embeddings_and_values`` grids differ, so we skip this analysis.
    print(
        "Skipping vars-to-dim CCA: it requires a single shared grid across regression "
        "tasks; per-task embeddings use different locations and sample sizes."
    )
    return {}, {}, reg_task_names


def run_dim_corr(task_names: list, task_types: dict, load_device: str, force=False):
    """
    Compute Pearson r between each embedding dimension and each regression label.
    Returns corr_matrices[emb_name] = (n_dims, n_tasks) array.
    """
    emb_keys = list(EMBEDDING_BACKENDS.keys())
    cache_path = DATA_DIR / "dim_label_correlations.npz"
    if cache_path.exists() and not force:
        print("Loading dim–label correlations from cache...")
        d = np.load(cache_path, allow_pickle=True)
        return {emb_name: d[emb_name] for emb_name in emb_keys}

    print("Computing dim–label correlations...")
    all_labels: dict = {}
    by_emb = {e: {} for e in emb_keys}
    for ds_name in task_names:
        _, bm, y, _, _ = load_embeddings_and_labels(ds_name, device=load_device, force=force)
        all_labels[ds_name] = (y, task_types[ds_name])
        for emb_name in emb_keys:
            by_emb[emb_name][ds_name] = bm[emb_name]

    corr_matrices = {}
    for emb_name in emb_keys:
        corr_matrices[emb_name] = compute_dim_label_correlations(
            by_emb[emb_name], all_labels, task_names
        )
        print(f"  {emb_name}: shape={corr_matrices[emb_name].shape}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **corr_matrices)
    print(f"Saved {cache_path}")

    return corr_matrices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCA interpretability for earth embeddings.")
    parser.add_argument("--cca",      action="store_true",
                        help="Variable-centric: which embedding dims project onto each variable?")
    parser.add_argument("--dim-cca",  action="store_true",
                        help="Dim-centric: which combination of env vars projects onto each embedding dim?")
    parser.add_argument("--dim-corr", action="store_true",
                        help="Dimension-centric: Pearson r matrix (which env vars does each dim track?)")
    parser.add_argument("--all",      action="store_true", help="Run all analyses")
    parser.add_argument("--rerun",    action="store_true", help="Recompute even if cached results exist")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    task_names = list(REGRESSION_DATASETS.keys()) + list(CLASSIFICATION_DATASETS.keys())
    task_types = {
        **{k: "regression" for k in REGRESSION_DATASETS},
        **{k: "classification" for k in CLASSIFICATION_DATASETS},
    }
    load_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CCA over {len(task_names)} datasets (lazy load per dataset); device={load_device}")

    fig_dir = FIGURES_DIR / "cca"

    if args.cca or args.all:
        cca_weights, cca_loadings, cca_corrs = run_cca(
            task_names, task_types, load_device, force=args.rerun
        )
        # Overview heatmap (all dims × all tasks) per component
        for comp in range(N_COMPONENTS):
            plot_cca_weight_heatmap(
                cca_loadings, task_names,
                component=comp,
                out_path=fig_dir / "cca_loadings_heatmap.png",
            )
        # Viz 1: variable-centric — which embedding dims project onto each variable?
        # Uses CCA weights (projection coefficients), one figure per variable.
        plot_variable_loadings(
            cca_weights, task_names,
            top_k=30, component=0,
            out_path=fig_dir / "variable_loadings" / "variable_loadings.png",
        )
        # Canonical correlations summary
        plot_canonical_correlations(
            cca_corrs, task_names,
            n_components=N_COMPONENTS,
            out_path=fig_dir / "cca_canonical_correlations.png",
        )

    if args.dim_cca or args.all:
        vars_to_dim_weights, vars_to_dim_corrs, reg_task_names = run_vars_to_dim_cca(
            task_names, task_types, force=args.rerun
        )
        if vars_to_dim_weights:
            plot_vars_to_dim_weights(
                vars_to_dim_weights, vars_to_dim_corrs, reg_task_names,
                top_k=30,
                out_path=fig_dir / "vars_to_dim" / "vars_to_dim.png",
            )

    if args.dim_corr or args.all:
        corr_matrices = run_dim_corr(
            task_names, task_types, load_device, force=args.rerun
        )
        plot_dim_label_correlations(
            corr_matrices, task_names,
            out_path=fig_dir / "dim_label_correlations" / "dim_label_corr.png",
        )

    print("\nDone.")

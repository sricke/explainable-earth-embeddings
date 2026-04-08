import sys
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, random_split

sys.path.append("..")
from prediction_datasets.datasets import load_dataset

from sklearn.linear_model import LassoLars, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_regression

REGRESSION_TASKS = ['air_temp', 'elevation', 'nightlights', 'pop_density', 'treecover']
CLASSIFICATION_TASKS = ['biome', 'ecoregion']#, 'countries']
BACKENDS = ['aef']

LASSO_MAX_SAMPLES = 10000  # LassoLars is O(n_samples * n_features), so we limit samples for speed. Can still rank dims with fewer samples.

# Speed: MI is k-NN–based and scales with n_samples; ~5k–10k is enough for ranking dims.
MI_MAX_SAMPLES = 5000
MI_N_NEIGHBORS = 2

def run_lasso_cv(x_train, y_train, x_test, y_test, alphas=[1.0, 0.1, 0.01, 0.001]):
    best_r2 = -np.inf
    best_selected_features = None
    best_coef = None
    best_intercept = None

    for alpha in alphas:
        results = run_lasso_feature_selection(x_train, y_train, x_test, y_test, alpha=alpha)
        r2, selected_features, coef, intercept = (results[k] for k in ["r2", "selected_features", "coef", "intercept"])
        if r2 > best_r2 + 0.01:  # Require at least 0.01 improvement to change best model (avoid overfitting to noise)
            best_r2 = r2
            best_selected_features = selected_features
            best_coef = coef
            best_intercept = intercept

    return {
        "r2": best_r2,
        "selected_features": best_selected_features,
        "coef": best_coef,
        "intercept": best_intercept,
    }


def run_lasso_feature_selection(x_train, y_train, x_test, y_test, alpha=0.01):
    if alpha is None:
        raise ValueError("Must provide alpha for Lasso regression")
    
    if x_train.shape[0] > LASSO_MAX_SAMPLES:
        rng = np.random.default_rng(0)
        idx = rng.choice(x_train.shape[0], size=LASSO_MAX_SAMPLES, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]

    model = LassoLars(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Lasso Feature Selection R2: {r2:.4f}")

    selected_features = np.where(model.coef_ != 0)[0]
    print(f"Selected features: {selected_features}")

    return {
        "r2": r2,
        "selected_features": selected_features,
        "coef": model.coef_,
        "intercept": model.intercept_,
    }

def run_l1_regularized_logistic_regression(x_train, y_train, x_test, y_test, C=10.0):
    model = LogisticRegression(penalty='l1', solver='saga', C=C, max_iter=1000)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"L1-Regularized Logistic Regression Accuracy: {acc:.4f}")

    selected_features = np.where(model.coef_[0] != 0)[0]
    print(f"Selected features: {selected_features}")

    return {
        "accuracy": acc,
        "selected_features": selected_features,
        "coef": model.coef_,
        "intercept": model.intercept_,
    }

def run_stability_selection(x_train, y_train, x_test, y_test, alpha=0.1, n_bootstraps=30):
    n_samples, n_features = x_train.shape
    feature_counts = np.zeros(n_features)

    for _ in range(n_bootstraps):
        indices = resample(np.arange(n_samples), replace=True)
        x_bootstrap = x_train[indices]
        y_bootstrap = y_train[indices]

        model = LassoLars(alpha=alpha)
        model.fit(x_bootstrap, y_bootstrap)
        feature_counts += (model.coef_ != 0).astype(int)

    stability_scores = feature_counts / n_bootstraps
    selected_features = np.where(stability_scores > 0.5)[0]  # Threshold can be adjusted
    print(f"Stability Selection - Selected features: {selected_features}")

    # Evaluate on test set using only selected features
    if len(selected_features) > 0:
        model_final = LassoLars(alpha=alpha)
        model_final.fit(x_train[:, selected_features], y_train)
        y_pred = model_final.predict(x_test[:, selected_features])
        r2 = r2_score(y_test, y_pred)
        print(f"Stability Selection R2 with selected features: {r2:.4f}")
    else:
        print("No features selected by stability selection.")

    return {
        "stability_scores": stability_scores,
        "selected_features": selected_features,
    }

def evaluate_mutual_information(x_train, y_train, x_test, y_test, max_samples=MI_MAX_SAMPLES):
    n = x_train.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        x_mi, y_mi = x_train[idx], y_train[idx]
    else:
        x_mi, y_mi = x_train, y_train
    mi_scores = mutual_info_regression(
        x_mi,
        y_mi,
        n_neighbors=MI_N_NEIGHBORS,
        random_state=0,
    )
    return mi_scores

def split_dataset(lat, lon, x, y, seed=42):
    coords = torch.tensor(np.column_stack([lon, lat]))
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = TensorDataset(coords, x, y)
    n = len(dataset)
    train_size = int(0.5 * n)
    train_set, test_set = random_split(
        dataset, [train_size, n - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    def _unpack(split):
        t = split.dataset.tensors
        idx = split.indices
        return t[0][idx], t[1][idx], t[2][idx]
    return _unpack(train_set), _unpack(test_set)

if __name__ == "__main__":
    from pathlib import Path
    from utils import (
        plot_importance_heatmap, plot_stability_heatmap,
        plot_mi_heatmap, plot_sparsity_barchart,
    )

    out_dir = Path("figures/feature_selection")

    lasso_coefs   = {b: [] for b in BACKENDS}
    stab_scores   = {b: [] for b in BACKENDS}
    mi_scores_all = {b: [] for b in BACKENDS}

    for task in REGRESSION_TASKS:
        for backend in BACKENDS:
            print(f"Running feature selection for {task} with {backend} embeddings")
            ds = load_dataset(task, backend, device='cpu')
            lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']

            (_, x_train, y_train), (_, x_test, y_test) = split_dataset(lat, lon, x, y)

            x_train = x_train.numpy().astype(np.float32, copy=False)
            y_train = y_train.numpy().astype(np.float32, copy=False)
            x_test  = x_test.numpy().astype(np.float32, copy=False)
            y_test  = y_test.numpy().astype(np.float32, copy=False)

            lasso_result = run_lasso_cv(x_train, y_train, x_test, y_test)
            stab_result  = run_stability_selection(x_train, y_train, x_test, y_test, alpha=0.1)
            mi           = evaluate_mutual_information(x_train, y_train, x_test, y_test)

            lasso_coefs[backend].append(np.abs(lasso_result['coef']))
            print(f"TOTAL NUMBER of selected features by Lasso: {(lasso_result['coef'] != 0).sum()}")
            stab_scores[backend].append(stab_result['stability_scores'])
            mi_scores_all[backend].append(mi)

    # Build (n_dims, n_tasks) matrices per backend
    importance_matrices = {}
    stability_freqs     = {}
    mi_matrices         = {}
    for backend in BACKENDS:
        coef_mat = np.column_stack(lasso_coefs[backend])          # (n_dims, n_tasks)
        col_max  = np.where(coef_mat.max(axis=0, keepdims=True) > 0,
                            coef_mat.max(axis=0, keepdims=True), 1.0)
        importance_matrices[backend] = {'lasso': coef_mat / col_max}
        stability_freqs[backend]     = np.column_stack(stab_scores[backend])
        mi_matrices[backend]         = np.column_stack(mi_scores_all[backend])

    plot_importance_heatmap(importance_matrices, REGRESSION_TASKS, method="lasso",
                            out_path=out_dir / "importance_heatmap.png")
    plot_stability_heatmap(stability_freqs, REGRESSION_TASKS,
                           out_path=out_dir / "stability_heatmap.png")
    plot_mi_heatmap(mi_matrices, REGRESSION_TASKS,
                    out_path=out_dir / "mi_heatmap.png")
    plot_sparsity_barchart(importance_matrices, REGRESSION_TASKS,
                           out_path=out_dir / "sparsity_barchart.png")
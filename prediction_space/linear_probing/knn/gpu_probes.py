"""
sklearn kNN probes for regression and classification.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold

import torch
import gc

K_VALUES = [1, 3, 5, 10]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def probe_knn_regression_gpu(X: np.ndarray, y: np.ndarray, k_values: list = K_VALUES,
                         cv: int = 5, seed: int = 42) -> dict[int, tuple[float, float]]:
    """kNN regression probe on GPU.

    Returns:
        dict mapping k -> (mean_r2, std_r2)
    """
    torch.manual_seed(seed)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    results = {}

    for k in k_values:
        fold_r2s = []

        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X_t[tr_idx], X_t[te_idx]
            y_tr, y_te = y_t[tr_idx], y_t[te_idx]

            # Standardize
            mu = X_tr.mean(0)
            sigma = X_tr.std(0, unbiased=False).clamp(min=1e-8)
            X_tr = (X_tr - mu) / sigma
            X_te = (X_te - mu) / sigma

            # Distances
            X2_te = (X_te**2).sum(dim=1, keepdim=True)
            X2_tr = (X_tr**2).sum(dim=1)
            dists = X2_te + X2_tr - 2 * (X_te @ X_tr.T)

            knn_idx = torch.topk(dists, k, largest=False).indices
            knn_y = y_tr[knn_idx]
            y_pred = knn_y.mean(dim=1)

            ss_res = ((y_te - y_pred) ** 2).sum()
            ss_tot = ((y_te - y_te.mean()) ** 2).sum()
            fold_r2s.append((1 - ss_res / ss_tot).item())

        scores = np.array(fold_r2s)
        results[k] = (scores.mean(), scores.std())

    del X_tr, X_te, y_tr, y_te
    del X2_tr, X2_te, dists
    del ss_res, ss_tot
    gc.collect()

    return results



def probe_knn_classification_gpu(X: np.ndarray, y: np.ndarray, k_values: list = K_VALUES,
                              cv: int = 5, seed: int = 42) -> dict[int, tuple[float, float]]:
    """kNN classification probe.

    Returns:
        dict mapping k -> (mean_accuracy, std_accuracy)
    """
    torch.manual_seed(seed)

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    y = LabelEncoder().fit_transform(y)

    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    results = {}

    for k in k_values:
        fold_scores = []

        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X_t[tr_idx], X_t[te_idx]
            y_tr, y_te = y_t[tr_idx], y_t[te_idx]

            # Standardize
            mu = X_tr.mean(0)
            sigma = X_tr.std(0, unbiased=False).clamp(min=1e-8)
            X_tr = (X_tr - mu) / sigma
            X_te = (X_te - mu) / sigma

            # Distances
            X2_te = (X_te**2).sum(dim=1, keepdim=True)
            X2_tr = (X_tr**2).sum(dim=1)
            dists = X2_te + X2_tr - 2 * (X_te @ X_tr.T)

            knn_idx = torch.topk(dists, k, largest=False).indices
            knn_y = y_tr[knn_idx]

            y_pred = torch.mode(knn_y, dim=1).values

            acc = (y_pred == y_te).float().mean().item()
            fold_scores.append(acc)

        scores = np.array(fold_scores)
        results[k] = (scores.mean(), scores.std())

    del X_tr, X_te, y_tr, y_te
    del X2_tr, X2_te, dists
    del acc
    gc.collect()

    return results
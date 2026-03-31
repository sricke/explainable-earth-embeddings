"""
GPU implementations of Ridge regression and Logistic Regression probes using PyTorch.

Both functions match the interface of their sklearn counterparts in ridge_probing.py:
    probe_ridge_gpu(X, y, cv, seed)      -> (mean_r2, std_r2)
    probe_logistic_gpu(X, y, cv, seed)   -> (mean_acc, std_acc)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Ridge — closed-form solution: w = (XᵀX + αI)⁻¹Xᵀy
# Cross-validates over a log-spaced grid of α values.
# ---------------------------------------------------------------------------


def probe_ridge_gpu(X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 42,
                    alphas=[0.1, 1.0, 10.0]) -> tuple[float, float]:
    """
    Ridge regression probe on GPU matching sklearn RidgeCV behavior.

    Per fold: selects best alpha via GCV (efficient LOO), then predicts on test.

    Args:
        X: embeddings, shape (N, d)
        y: regression targets, shape (N,)
        cv: number of CV folds
        seed: random seed for fold splitting
        alphas: iterable of regularisation strengths (default: sklearn's (0.1, 1.0, 10.0))

    Returns:
        (mean_r2, std_r2) across folds
    """
    torch.manual_seed(seed)
    alphas = torch.tensor(alphas, dtype=torch.float64, device=DEVICE)

    # Filter NaNs / non-finite
    mask = np.isfinite(X).all(axis=1) & ~np.isnan(X).any(axis=1) & np.isfinite(y) & ~np.isnan(y)
    X, y = X[mask], y[mask]

    X_t = torch.tensor(X, dtype=torch.float64, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float64, device=DEVICE)

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_r2s = []

    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X_t[tr_idx], X_t[te_idx]
        y_tr, y_te = y_t[tr_idx], y_t[te_idx]

        # Standardize using training stats (population std, matching sklearn StandardScaler)
        mu = X_tr.mean(0)
        sigma = X_tr.std(0, unbiased=False).clamp(min=1e-8)
        X_tr_std = (X_tr - mu) / sigma
        X_te_std = (X_te - mu) / sigma

        # Center y — matches sklearn RidgeCV(fit_intercept=True) which centers y internally
        mu_y = y_tr.mean()
        y_tr_c = y_tr - mu_y

        # SVD-based GCV for per-fold alpha selection (matches sklearn _RidgeGCV)
        # sklearn uses: LOO error_i = c_i / diag(G^-1)_i  where G = XX^T + αI, c = G^-1 y
        # Equivalently via hat matrix: LOO error_i = residual_i / (1 - h_ii)
        # where h_ii = sum_j u_ij^2 * s_j^2/(s_j^2 + α)  (diagonal of H = U diag(d) U^T)
        U, s, _ = torch.linalg.svd(X_tr_std, full_matrices=False)
        UTy = U.T @ y_tr_c

        best_gcv = float('inf')
        best_alpha = alphas[0]

        for alpha in alphas:
            d = s ** 2 / (s ** 2 + alpha)          # (k,)  hat diagonal factors
            y_hat = U @ (d * UTy)                   # (n,)  fitted values on centered y
            residuals = y_tr_c - y_hat              # (n,)
            leverage = (U ** 2) @ d                 # (n,)  diagonal of hat matrix; < 1 since d < 1
            gcv = ((residuals / (1 - leverage)) ** 2).mean()
            if gcv.item() <= best_gcv:
                best_gcv = gcv.item()
                best_alpha = alpha

        # Fit with best alpha on centered y; intercept = mu_y (X is already zero-mean)
        XtX = X_tr_std.T @ X_tr_std
        Xty = X_tr_std.T @ y_tr_c
        I = torch.eye(XtX.shape[0], device=DEVICE)
        w = torch.linalg.solve(XtX + best_alpha * I, Xty)
        y_pred = X_te_std @ w + mu_y

        ss_res = ((y_te - y_pred) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum()
        fold_r2s.append((1 - ss_res / ss_tot).item())

    fold_r2s = np.array(fold_r2s)
    return fold_r2s.mean(), fold_r2s.std()


# ---------------------------------------------------------------------------
# Logistic Regression — SGD with cross-entropy loss
# ---------------------------------------------------------------------------

class _LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def probe_logistic_gpu(X: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 42,
                       lr: float = 0.1, epochs: int = 200, weight_decay: float = 1e-4,
                       batch_size: int = 4096) -> tuple[float, float]:
    """Logistic regression probe on GPU via mini-batch SGD.

    Args:
        X: embeddings, shape (N, d)
        y: integer class labels, shape (N,)
        cv: number of CV folds
        seed: random seed
        lr: learning rate
        epochs: training epochs per fold
        weight_decay: L2 regularisation (equivalent to C=1/weight_decay in sklearn)
        batch_size: mini-batch size

    Returns:
        (mean_accuracy, std_accuracy) across folds
    """
    torch.manual_seed(seed)

    mask = np.isfinite(X).all(axis=1) & ~np.isnan(X).any(axis=1) & np.isfinite(y) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    y = LabelEncoder().fit_transform(y.astype(int))
    n_classes = int(y.max()) + 1

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_scores = []

    for tr_idx, te_idx in kf.split(X):
        X_tr = torch.tensor(X[tr_idx], dtype=torch.float64, device=DEVICE)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long,    device=DEVICE)
        X_te = torch.tensor(X[te_idx], dtype=torch.float64, device=DEVICE)
        y_te = torch.tensor(y[te_idx], dtype=torch.long,    device=DEVICE)

        # Standardise
        mu, sigma = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma

        model = _LinearClassifier(X_tr.shape[1], n_classes).to(DEVICE, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        n = X_tr.shape[0]
        for _ in tqdm(range(epochs), desc="Training Epochs"):
            perm = torch.randperm(n, device=DEVICE)
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_tr[idx]), y_tr[idx])
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = model(X_te).argmax(dim=1)
            acc = (preds == y_te).float().mean().item()
        fold_scores.append(acc)

    scores = np.array(fold_scores)
    return scores.mean(), scores.std()

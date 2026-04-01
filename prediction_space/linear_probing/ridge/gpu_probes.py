"""
GPU implementations of Ridge regression and Logistic Regression probes using PyTorch.

    probe_ridge_gpu(X_train, y_train, X_test, y_test, ...) -> (test_r2, cv_std, alpha_info)
    probe_logistic_gpu(X, y, cv, seed)   -> (mean_acc, std_acc)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ridge_val_r2(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    alpha: float,
) -> float:
    """Standardize on training fold, fit ridge (intercept via y mean), R² on validation."""
    mu_x = X_tr.mean(0)
    sig_x = X_tr.std(0, unbiased=False).clamp(min=1e-8)
    X_tr_s = (X_tr - mu_x) / sig_x
    X_va_s = (X_va - mu_x) / sig_x
    mu_y = y_tr.mean()
    y_c = y_tr - mu_y
    d = X_tr_s.shape[1]
    XtX = X_tr_s.T @ X_tr_s
    Xty = X_tr_s.T @ y_c
    dev = X_tr_s.device
    I = torch.eye(d, device=dev, dtype=X_tr_s.dtype)
    w = torch.linalg.solve(XtX + alpha * I, Xty)
    pred_va = X_va_s @ w + mu_y
    ss_res = ((y_va - pred_va) ** 2).sum()
    ss_tot = ((y_va - y_va.mean()) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    return (1 - ss_res / ss_tot).item()


def probe_ridge_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv: int = 5,
    seed: int = 42,
    alphas=None,
    device: torch.device | None = None,
) -> tuple[float, float, dict]:
    """
    Choose ``alpha`` by K-fold CV R² on the training set (GPU tensor ops), refit on all
    training rows, return coefficient of determination on the held-out test set.
    """
    dev = device if device is not None else DEVICE
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    alphas_list = [float(a) for a in alphas]

    m_tr = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    m_te = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
    X_tr_np, y_tr_np = X_train[m_tr], y_train[m_tr]
    X_te_np, y_te_np = X_test[m_te], y_test[m_te]
    if X_tr_np.shape[0] < cv:
        raise ValueError(
            f"Training rows after filtering ({X_tr_np.shape[0]}) must be >= cv folds ({cv})"
        )
    if X_te_np.shape[0] < 1:
        raise ValueError("Need at least one finite test row for evaluation")

    torch.manual_seed(seed)
    X_tr = torch.tensor(X_tr_np, dtype=torch.float64, device=dev)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float64, device=dev)
    X_te = torch.tensor(X_te_np, dtype=torch.float64, device=dev)
    y_te = torch.tensor(y_te_np, dtype=torch.float64, device=dev)

    n_tr = X_tr.shape[0]
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    mean_cv_r2: dict[float, float] = {}
    std_cv_r2: dict[float, float] = {}

    for alpha in alphas_list:
        fold_r2 = []
        for sub_tr, sub_va in kf.split(np.arange(n_tr)):
            r2 = _ridge_val_r2(
                X_tr[sub_tr], y_tr[sub_tr], X_tr[sub_va], y_tr[sub_va], alpha
            )
            fold_r2.append(r2)
        mean_cv_r2[alpha] = float(np.mean(fold_r2))
        std_cv_r2[alpha] = float(np.std(fold_r2))

    best_alpha = max(mean_cv_r2, key=mean_cv_r2.get)
    test_r2 = _ridge_val_r2(X_tr, y_tr, X_te, y_te, best_alpha)

    alpha_info = {
        "mean_cv_r2": mean_cv_r2,
        "std_cv_r2": std_cv_r2,
        "best_alpha": best_alpha,
        "selection_counts": {a: cv if a == best_alpha else 0 for a in mean_cv_r2},
    }
    return float(test_r2), std_cv_r2[best_alpha], alpha_info


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
                       batch_size: int = 4096,
                       device: torch.device | None = None) -> tuple[float, float]:
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
        device: torch device to use (defaults to module-level DEVICE)

    Returns:
        (mean_accuracy, std_accuracy) across folds
    """
    dev = device if device is not None else DEVICE
    torch.manual_seed(seed)

    mask = np.isfinite(X).all(axis=1) & ~np.isnan(X).any(axis=1) & np.isfinite(y) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    y = LabelEncoder().fit_transform(y.astype(int))
    n_classes = int(y.max()) + 1

    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    fold_scores = []

    for tr_idx, te_idx in kf.split(X):
        X_tr = torch.tensor(X[tr_idx], dtype=torch.float64, device=dev)
        y_tr = torch.tensor(y[tr_idx], dtype=torch.long,    device=dev)
        X_te = torch.tensor(X[te_idx], dtype=torch.float64, device=dev)
        y_te = torch.tensor(y[te_idx], dtype=torch.long,    device=dev)

        # Standardise
        mu, sigma = X_tr.mean(0), X_tr.std(0).clamp(min=1e-8)
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma

        model = _LinearClassifier(X_tr.shape[1], n_classes).to(dev, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        n = X_tr.shape[0]
        for _ in tqdm(range(epochs), desc="Training Epochs"):
            perm = torch.randperm(n, device=dev)
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

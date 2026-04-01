"""
GPU-accelerated feature selection methods using PyTorch (torch.float64).

Methods
-------
fit_sparse_regression_gpu   — LassoCV + ElasticNetCV (coordinate descent, cross-validated alpha)
fit_sparse_classification   — L1 / ElasticNet logistic regression
stability_selection         — bootstrap LASSO selection frequencies
compute_mi                  — mutual information per dimension (k-NN KSG estimator)
mrmr                        — greedy min-Redundancy Max-Relevance selection
progressive_ablation        — performance vs. fraction of top dims removed
"""

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(arr, dtype=DTYPE, device=device)


def _finite_mask(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(X).all(axis=1) & np.isfinite(y)

def _lipschitz(X: torch.Tensor) -> float:
    """Lipschitz constant for logistic regression gradient: largest eigenvalue of X^T X / (4n)."""
    n = X.shape[0]
    with torch.no_grad():
        cov = (X.T @ X) / n
        eigs = torch.linalg.eigvalsh(cov)
        return float(eigs.max().real) / 4.0


def _standardize(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Z-score standardize. Returns (X_scaled, mean, std); std clamped to avoid div-by-zero."""
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False).clamp(min=1e-12)
    return (X - mean) / std, mean, std


def _soft_threshold(x, lam):
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

def _max_eigval_power(G, n_iter=50):
    v = torch.randn(G.shape[0], device=G.device, dtype=G.dtype)
    v = v / v.norm()
    for _ in range(n_iter):
        v = G @ v
        v = v / v.norm()
    return torch.dot(v, G @ v)


def _kfold_indices(n: int, k: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return (train_idx, val_idx) pairs for k-fold CV (sequential splits, no shuffle)."""
    fold_size = n // k
    splits = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        val_idx = torch.arange(val_start, val_end, device=device)
        train_idx = torch.cat([
            torch.arange(0, val_start, device=device),
            torch.arange(val_end, n, device=device),
        ])
        splits.append((train_idx, val_idx))
    return splits


# ---------------------------------------------------------------------------
# FISTA solvers (proximal gradient with Nesterov momentum)
# ---------------------------------------------------------------------------

def _lasso_fista(X, y, alpha, w_init=None,
                 max_iter=1000, tol=1e-7,
                 G=None, c=None):

    n, d = X.shape
    if G is None:
        G = X.T @ X / n
    if c is None:
        c = X.T @ y / n

    L = float(_max_eigval_power(G).clamp(min=1e-12))
    step = 1.0 / L

    w = torch.zeros(d, dtype=X.dtype, device=X.device) if w_init is None else w_init.clone()
    z = w.clone()
    t = 1.0

    for _ in range(max_iter):
        grad = G @ z - c
        w_new = _soft_threshold(z - step * grad, alpha * step)

        if torch.norm(w_new - w) <= tol * max(1.0, torch.norm(w)):
            return w_new

        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = w_new + ((t - 1) / t_new) * (w_new - w)

        w = w_new
        t = t_new

    return w


def _elasticnet_fista(X, y, alpha, l1_ratio=0.5,
                     max_iter=1000, tol=1e-7,
                     w_init=None, G=None, c=None):

    n, d = X.shape
    alpha_l1 = alpha * l1_ratio
    alpha_l2 = alpha * (1 - l1_ratio)

    if G is None:
        G = X.T @ X / n
    if c is None:
        c = X.T @ y / n

    L = float(_max_eigval_power(G).clamp(min=1e-12)) + alpha_l2
    step = 1.0 / L

    w = torch.zeros(d, dtype=X.dtype, device=X.device) if w_init is None else w_init.clone()
    z = w.clone()
    t = 1.0

    for _ in range(max_iter):
        grad = G @ z - c + alpha_l2 * z
        w_new = _soft_threshold(z - step * grad, alpha_l1 * step)

        if torch.norm(w_new - w) <= tol * max(1.0, torch.norm(w)):
            return w_new

        t_new = 0.5 * (1 + (1 + 4 * t * t) ** 0.5)
        z = w_new + ((t - 1) / t_new) * (w_new - w)

        w = w_new
        t = t_new

    return w


def _ridge_solve(X: torch.Tensor, y: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Closed-form Ridge: (X^T X + alpha*n*I)^{-1} X^T y"""
    n, d = X.shape
    A = X.T @ X + alpha * n * torch.eye(d, dtype=DTYPE, device=X.device)
    return torch.linalg.solve(A, X.T @ y)


def _logistic_prox_grad(X: torch.Tensor, y_oh: torch.Tensor,
                         alpha_l1: float, alpha_l2: float,
                         max_iter: int = 1000, tol: float = 1e-6) -> torch.Tensor:
    """
    Proximal gradient for logistic regression with elastic-net penalty.

    Args:
        y_oh: one-hot targets, shape (n, n_classes)
    Returns:
        W: shape (n_classes, d)
    """
    n, d = X.shape
    n_classes = y_oh.shape[1]
    W = torch.zeros(n_classes, d, dtype=DTYPE, device=X.device)
    step = 1.0 / (_lipschitz(X) + alpha_l2)
    for _ in range(max_iter):
        logits = X @ W.T                               # (n, n_classes)
        probs = torch.softmax(logits, dim=1)           # (n, n_classes)
        grad = (probs - y_oh).T @ X / n + alpha_l2 * W  # (n_classes, d)
        W_new = _soft_threshold(W - step * grad, alpha_l1 * step)
        if torch.norm(W_new - W) < tol:
            return W_new
        W = W_new
    return W


# ---------------------------------------------------------------------------
# Cross-validated alpha selection for logistic regression (classification)
# ---------------------------------------------------------------------------

def _cv_alpha_logistic(
    X: torch.Tensor, y_oh: torch.Tensor, alphas: torch.Tensor,
    cv: int = 3, max_iter: int = 500,
) -> tuple:
    """Cross-validate L1-logistic over alphas; select by minimum val cross-entropy.

    Returns:
        best_alpha  — tensor
        alpha_loss  — dict {alpha_float: mean_val_loss}
    """
    folds = _kfold_indices(len(y_oh), cv, X.device)
    alpha_keys = [float(a.item()) for a in alphas]
    alpha_loss_per_fold = {a: [] for a in alpha_keys}

    for train_idx, val_idx in folds:
        Xtr, ytr_oh = X[train_idx], y_oh[train_idx]
        Xval, yval_oh = X[val_idx], y_oh[val_idx]

        Xtr_s, mean, std = _standardize(Xtr)
        Xval_s = (Xval - mean) / std

        for alpha in alphas:
            W = _logistic_prox_grad(Xtr_s, ytr_oh,
                                    alpha_l1=float(alpha.item()), alpha_l2=0.0,
                                    max_iter=max_iter)
            logits = Xval_s @ W.T                         # (n_val, n_classes)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(yval_oh * log_probs).sum(dim=1).mean().item()
            alpha_loss_per_fold[float(alpha.item())].append(loss)

    alpha_mean_loss = {a: float(np.nanmean(v)) for a, v in alpha_loss_per_fold.items()}
    best_alpha_val = min(alpha_mean_loss, key=lambda a: alpha_mean_loss[a])
    best_alpha = alphas[alpha_keys.index(best_alpha_val)]
    return best_alpha, alpha_mean_loss


# ---------------------------------------------------------------------------
# Cross-validated alpha selection for regression models
# ---------------------------------------------------------------------------

def _cv_alpha_lasso(X: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor,
                    cv: int = 3, max_iter: int = 1000) -> float:

    folds = _kfold_indices(len(y), cv, X.device)
    d = X.shape[1]
    alpha_keys = [float(a.item()) for a in alphas]
    alpha_mse_per_fold = {a: [] for a in alpha_keys}

    for train_idx, val_idx in folds:
        Xtr, ytr = X[train_idx], y[train_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Standardize
        Xtr_s, mean, std = _standardize(Xtr)
        Xval_s = (Xval - mean) / std

        # Center y
        ytr_c = ytr - ytr.mean()

        # Precompute Gram matrix once per fold; reused across the warm-start alpha path
        G = Xtr_s.T @ Xtr_s / len(train_idx)
        c = Xtr_s.T @ ytr_c / len(train_idx)

        w = torch.zeros(d, dtype=DTYPE, device=X.device)  # reset per fold

        for alpha in alphas:   # warm start path
            w = _lasso_fista(Xtr_s, ytr_c, alpha, max_iter=max_iter, w_init=w, G=G, c=c)
            pred = Xval_s @ w + ytr.mean()
            mse_fold = ((yval - pred) ** 2).mean().item()
            alpha_mse_per_fold[float(alpha.item())].append(mse_fold)

    y_var = y.var().item()
    y_var = max(y_var, 1e-12)  # avoid div-by-zero in case of constant y
    alpha_mean_mse = {a: float(np.nanmean(mses)) for a, mses in alpha_mse_per_fold.items()}
    alpha_std_mse  = {a: float(np.nanstd(mses))  for a, mses in alpha_mse_per_fold.items()}
    alpha_r2       = {a: float(1.0 - mse / y_var) for a, mse in alpha_mean_mse.items()}

    best_alpha_val = min(alpha_mean_mse, key=lambda a: alpha_mean_mse[a])
    best_alpha = alphas[alpha_keys.index(best_alpha_val)]
    best_mse = alpha_mean_mse[best_alpha_val]
    best_r2 = 1.0 - best_mse / y_var
    return best_alpha, best_r2, alpha_r2, alpha_mean_mse, alpha_std_mse

def _cv_alpha_enet(X: torch.Tensor, y: torch.Tensor, alphas: torch.Tensor,
                   l1_ratio: float = 0.5, cv: int = 3, max_iter: int = 1000) -> float:
    folds = _kfold_indices(len(y), cv, X.device)
    d = X.shape[1]
    alpha_keys = [float(a.item()) for a in alphas]
    alpha_mse_per_fold = {a: [] for a in alpha_keys}

    for train_idx, val_idx in folds:
        Xtr, ytr = X[train_idx], y[train_idx]
        Xval, yval = X[val_idx], y[val_idx]

        # Standardize
        Xtr_s, mean, std = _standardize(Xtr)
        Xval_s = (Xval - mean) / std

        # Center y
        ytr_c = ytr - ytr.mean()

        # Precompute Gram matrix once per fold; reused across the warm-start alpha path
        G = Xtr_s.T @ Xtr_s / len(train_idx)
        c = Xtr_s.T @ ytr_c / len(train_idx)

        w = torch.zeros(d, dtype=DTYPE, device=X.device)  # reset per fold

        for alpha in alphas:   # warm start path
            w = _elasticnet_fista(Xtr_s, ytr_c, alpha, l1_ratio=l1_ratio, max_iter=max_iter, w_init=w, G=G, c=c)
            pred = Xval_s @ w + ytr.mean()
            mse_fold = ((yval - pred) ** 2).mean().item()
            alpha_mse_per_fold[float(alpha.item())].append(mse_fold)

    y_var = y.var().item()
    y_var = max(y_var, 1e-12)  # avoid div-by-zero if y is constant
    alpha_mean_mse = {a: float(np.nanmean(mses)) for a, mses in alpha_mse_per_fold.items()}
    alpha_std_mse  = {a: float(np.nanstd(mses))  for a, mses in alpha_mse_per_fold.items()}
    alpha_r2       = {a: float(1.0 - mse / y_var) for a, mse in alpha_mean_mse.items()}
    
    best_alpha_val = min(alpha_mean_mse, key=lambda a: alpha_mean_mse[a])
    best_alpha = alphas[alpha_keys.index(best_alpha_val)]
    best_mse = alpha_mean_mse[best_alpha_val]
    best_r2 = 1.0 - best_mse / y_var
    return best_alpha, best_r2, alpha_r2, alpha_mean_mse, alpha_std_mse


# ---------------------------------------------------------------------------
# Mutual information: k-NN KSG estimator on GPU
# ---------------------------------------------------------------------------

def _knn_mi_gpu(x: torch.Tensor, z: torch.Tensor, k: int = 3,
                chunk_size: int = 256) -> float:
    """
    Estimate MI(x; z) between two 1-D tensors using the KSG estimator.
    Uses Chebyshev distance in joint space and strict ball counts in marginals.
    Distances are computed in row-chunks of size `chunk_size` to bound GPU memory
    to O(chunk_size × n) instead of O(n²).
    """
    n = x.shape[0]
    joint = torch.stack([x, z], dim=1)   # (n, 2)

    eps = torch.zeros(n, dtype=DTYPE, device=x.device)
    nx  = torch.zeros(n, dtype=DTYPE, device=x.device)
    nz  = torch.zeros(n, dtype=DTYPE, device=x.device)

    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        rows = slice(start, end)
        chunk_len = end - start
        diag = torch.arange(chunk_len, device=x.device)

        # Joint Chebyshev distances for this chunk vs all points: (chunk, n, 2)
        diff = joint[rows].unsqueeze(1) - joint.unsqueeze(0)
        dist_j = diff.abs().amax(dim=2)              # (chunk, n)
        dist_j[diag, start + diag] = float("inf")   # exclude self

        topk_vals, _ = dist_j.topk(k, dim=1, largest=False)
        eps[rows] = topk_vals[:, -1]                 # k-th NN distance

        # Marginal distances
        dist_x = (x[rows].unsqueeze(1) - x.unsqueeze(0)).abs()  # (chunk, n)
        dist_z = (z[rows].unsqueeze(1) - z.unsqueeze(0)).abs()  # (chunk, n)
        dist_x[diag, start + diag] = float("inf")
        dist_z[diag, start + diag] = float("inf")

        eps_row = eps[rows].unsqueeze(1)             # (chunk, 1)
        nx[rows] = (dist_x < eps_row).sum(dim=1).to(DTYPE)
        nz[rows] = (dist_z < eps_row).sum(dim=1).to(DTYPE)

    psi = torch.special.digamma
    mi = float(
        psi(torch.tensor(k, dtype=DTYPE, device=x.device))
        + psi(torch.tensor(n, dtype=DTYPE, device=x.device))
        - psi(nx + 1).mean()
        - psi(nz + 1).mean()
    )
    return max(mi, 0.0)


def _mi_all_dims(X: torch.Tensor, y: torch.Tensor, k: int = 3) -> torch.Tensor:
    """MI between every column of X and y."""
    d = X.shape[1]
    mi = torch.zeros(d, dtype=DTYPE, device=X.device)
    for j in range(d):
        mi[j] = _knn_mi_gpu(X[:, j], y, k=k)
    return mi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_sparse_regression_gpu(
    X: np.ndarray, y: np.ndarray,
    device: str | torch.device = "cuda",
    cv: int = 5, alphas: torch.Tensor = torch.Tensor([0.1, 1.0, 10.0]),
    n_alphas: int = 10, max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit LassoCV and ElasticNetCV on GPU (coordinate descent with cross-validated alpha).

    Returns:
        (lasso_coef, enet_coef) — shape (n_dims,)
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X, y = X[mask], y[mask]

    Xt = _to_tensor(X, device)
    yt = _to_tensor(y, device)
    Xs, _, _ = _standardize(Xt) # only use for final lasso or enet coefficients, not for CV splits (to avoid data leakage)

    print("    Finding best alpha via cross-validation...")
    if alphas is None:
        lam_max = (Xs.T @ yt).abs().max().item() / Xs.shape[0] # data leakage but right now only care about coefficients. In principle could compute separate lam_max per CV fold to avoid leakage, but would be more expensive and likely not change results much.
        alphas = torch.logspace(np.log10(lam_max * 1e-4), np.log10(lam_max), n_alphas, device=device)

    yt_c = yt - yt.mean()

    best_lasso_alpha, lasso_r2, lasso_alpha_r2, lasso_alpha_mse, lasso_alpha_std_mse = _cv_alpha_lasso(Xt, yt_c, alphas, cv=cv, max_iter=max_iter)
    lasso_coef = _lasso_fista(Xs, yt_c, best_lasso_alpha, max_iter=max_iter).cpu().numpy()

    print(f"    Lasso alpha: {best_lasso_alpha:.4e}, nonzero features: {(lasso_coef != 0).sum()}.")
    print(f"    Lasso CV R^2: {lasso_r2:.4f}")

    best_enet_alpha, enet_r2, enet_alpha_r2, enet_alpha_mse, enet_alpha_std_mse = _cv_alpha_enet(Xt, yt_c, alphas, cv=cv, max_iter=max_iter)
    enet_coef = _elasticnet_fista(Xs, yt_c, best_enet_alpha, l1_ratio=0.5, max_iter=max_iter).cpu().numpy()

    print(f"    Elastic Net alpha: {best_enet_alpha:.4e}, nonzero features: {(enet_coef != 0).sum()}")
    print(f"    Elastic Net CV R^2: {enet_r2:.4f}")

    cv_info = {
        "lasso": {"best_alpha": float(best_lasso_alpha), "cv_r2": lasso_r2,
                  "alpha_r2": lasso_alpha_r2, "alpha_cv_mse": lasso_alpha_mse, "alpha_cv_std_mse": lasso_alpha_std_mse},
        "enet":  {"best_alpha": float(best_enet_alpha),  "cv_r2": enet_r2,
                  "alpha_r2": enet_alpha_r2,  "alpha_cv_mse": enet_alpha_mse,  "alpha_cv_std_mse": enet_alpha_std_mse},
    }
    return lasso_coef, enet_coef, cv_info


def fit_sparse_classification_gpu(
    X: np.ndarray, y: np.ndarray,
    device: str | torch.device = "cuda",
    cv: int = 3,  alphas: torch.Tensor = torch.Tensor([0.1, 1.0, 10.0]), 
    n_alphas: int = 10, max_iter: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit L1 and Elastic Net logistic regression on GPU with CV-selected alpha.

    Returns:
        (l1_coef, enet_coef) — shape (n_classes, n_dims)
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X = X[mask]
    y_enc = LabelEncoder().fit_transform(y[mask].astype(int))
    n_classes = int(y_enc.max()) + 1

    Xt = _to_tensor(X, device)
    Xs, _, _ = _standardize(Xt)

    y_oh = torch.zeros(len(y_enc), n_classes, dtype=DTYPE, device=device)
    y_oh.scatter_(1, torch.tensor(y_enc, device=device).unsqueeze(1), 1.0)

    if alphas is None:
        # lam_max: max |gradient of cross-entropy at W=0| = max |(1/K - y_oh).T @ Xs / n|
        n = Xs.shape[0]
        grad_at_zero = ((1.0 / n_classes) - y_oh).T @ Xs / n
        lam_max = float(grad_at_zero.abs().max().item())
        alphas = torch.logspace(np.log10(lam_max * 1e-4), np.log10(lam_max), n_alphas, device=device)

    print("    Finding best alpha via cross-validation (logistic)...")
    best_l1_alpha, _ = _cv_alpha_logistic(Xt, y_oh, alphas, cv=cv, max_iter=max_iter // 2)
    l1_W = _logistic_prox_grad(Xs, y_oh, alpha_l1=float(best_l1_alpha.item()), alpha_l2=0.0, max_iter=max_iter)
    print(f"    L1 alpha: {float(best_l1_alpha):.4e}, nonzero features: {(l1_W != 0).any(dim=0).sum().item()}")

    best_enet_alpha, _ = _cv_alpha_logistic(Xt, y_oh, alphas, cv=cv, max_iter=max_iter // 2)
    enet_W = _logistic_prox_grad(Xs, y_oh, alpha_l1=float(best_enet_alpha.item()) * 0.5,
                                  alpha_l2=float(best_enet_alpha.item()) * 0.5, max_iter=max_iter)
    print(f"    Enet alpha: {float(best_enet_alpha):.4e}, nonzero features: {(enet_W != 0).any(dim=0).sum().item()}")

    return l1_W.cpu().numpy(), enet_W.cpu().numpy()


def stability_selection_gpu(
    X: np.ndarray, y: np.ndarray, task_type: str,
    device: str | torch.device = "cuda",
    n_bootstrap: int = 50, subsample_frac: float = 0.5,
    alpha_frac: float = 0.1,
) -> np.ndarray:
    """Bootstrap subsampled LASSO to estimate selection frequency per dimension.

    Returns:
        selection_freq — shape (n_dims,), values in [0, 1]
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X, y = X[mask], y[mask]
    if task_type == "classification":
        y = LabelEncoder().fit_transform(y.astype(int)).astype(float)

    n, d = X.shape
    n_sub = int(n * subsample_frac)
    counts = torch.zeros(d, dtype=DTYPE, device=device)

    Xt = _to_tensor(X, device)
    yt = _to_tensor(y, device)

    # For classification: CV over fixed candidates to find best alpha once,
    # then reuse that alpha for all bootstrap iterations.
    cls_alpha = None
    if task_type == "classification":
        # y is already label-encoded (0..n_classes-1) from the LabelEncoder above
        n_classes = int(y.max()) + 1
        y_oh_full = torch.zeros(n, n_classes, dtype=DTYPE, device=device)
        y_oh_full.scatter_(1, torch.tensor(y.astype(int), device=device).unsqueeze(1), 1.0)
        candidate_alphas = torch.tensor([0.1, 10.0, 100.0], dtype=DTYPE, device=device)
        cls_alpha_t, _ = _cv_alpha_logistic(Xt, y_oh_full, candidate_alphas,
                                            cv=3, max_iter=300)
        cls_alpha = float(cls_alpha_t.item())
        print(f"    Stability (classification) best alpha: {cls_alpha:.4e}")

    for _ in range(n_bootstrap):
        idx = torch.randperm(n, device=device)[:n_sub]
        Xb, yb = Xt[idx], yt[idx]
        Xb_s, _, _ = _standardize(Xb)

        if task_type == "regression":
            alpha = float(yb.std()) * alpha_frac
            coef = _lasso_fista(Xb_s, yb, alpha)
        else:
            n_classes = int(yt.max().item()) + 1
            y_oh = torch.zeros(n_sub, n_classes, dtype=DTYPE, device=device)
            y_oh.scatter_(1, yb.long().unsqueeze(1), 1.0)
            W = _logistic_prox_grad(Xb_s, y_oh, alpha_l1=cls_alpha,
                                    alpha_l2=0.0, max_iter=500)
            coef = W.abs().max(dim=0).values

        counts += (coef.abs() > 1e-6).to(DTYPE)

    return (counts / n_bootstrap).cpu().numpy()


def compute_mi_gpu(
    X: np.ndarray, y: np.ndarray, task_type: str,
    device: str | torch.device = "cuda",
    n_neighbors: int = 3,
    max_samples: int = 10_000,
) -> np.ndarray:
    """Compute mutual information between each dimension and the target (k-NN KSG, GPU).

    Returns:
        mi — shape (n_dims,)
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X, y = X[mask], y[mask]

    # Subsample to avoid OOM from O(chunk_size * n) pairwise distances
    if len(y) > max_samples:
        idx = np.random.default_rng(42).choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]

    Xt = _to_tensor(X, device)
    yt = _to_tensor(y, device)
    Xs, _, _ = _standardize(Xt)

    return _mi_all_dims(Xs, yt, k=n_neighbors).cpu().numpy()


def mrmr_gpu(
    X: np.ndarray, y: np.ndarray, task_type: str,
    device: str | torch.device = "cuda",
    n_features: int = 32, n_neighbors: int = 3,
    max_samples: int = 10_000,
) -> list[int]:
    """Greedy min-Redundancy Max-Relevance feature selection (GPU).

    Returns:
        selected — list of dim indices in selection order
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X, y = X[mask], y[mask]

    # Subsample to avoid OOM from O(chunk_size * n) pairwise distances in MI
    if len(y) > max_samples:
        idx = np.random.default_rng(42).choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]

    Xt = _to_tensor(X, device)
    yt = _to_tensor(y, device)
    Xs, _, _ = _standardize(Xt)

    relevance = _mi_all_dims(Xs, yt, k=n_neighbors)
    selected: list[int] = []
    remaining = list(range(Xs.shape[1]))

    for _ in range(n_features):
        if not remaining:
            break
        rem_t = torch.tensor(remaining, device=device)
        if not selected:
            best = remaining[int(relevance[rem_t].argmax().item())]
        else:
            redundancy = torch.tensor(
                [
                    np.mean([_knn_mi_gpu(Xs[:, s], Xs[:, f], k=n_neighbors) for s in selected])
                    for f in remaining
                ],
                dtype=DTYPE, device=device,
            )
            best = remaining[int((relevance[rem_t] - redundancy).argmax().item())]
        selected.append(best)
        remaining.remove(best)

    return selected


def progressive_ablation_gpu(
    X: np.ndarray, y: np.ndarray, task_type: str,
    importance_scores: np.ndarray,
    device: str | torch.device = "cuda",
    steps: int = 10, cv: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Progressively remove the most-important dims and track CV performance (GPU).

    Regression scoring: R^2.  Classification scoring: accuracy.

    Returns:
        (fractions, scores) — fractions removed (0..0.9), CV score at each step
    """
    device = torch.device(device)
    mask = _finite_mask(X, y)
    X, y = X[mask], y[mask]

    ranked_dims = np.argsort(importance_scores)[::-1]
    n_dims = X.shape[1]
    fractions = np.linspace(0, 0.9, steps)
    scores = []

    Xt = _to_tensor(X, device)

    if task_type == "classification":
        y_enc = LabelEncoder().fit_transform(y.astype(int))
        yt = torch.tensor(y_enc, dtype=torch.long, device=device)
        n_classes = int(yt.max().item()) + 1
    else:
        yt = _to_tensor(y, device)
        n_classes = None

    folds = _kfold_indices(len(yt), cv, device)

    for frac in fractions:
        n_remove = int(frac * n_dims)
        keep = ranked_dims[n_remove:]
        Xk = Xt[:, keep]

        fold_scores = []
        for train_idx, val_idx in folds:
            Xtr, Xval = Xk[train_idx], Xk[val_idx]
            Xtr_s, mean, std = _standardize(Xtr)
            Xval_s = (Xval - mean) / std
            ytr, yval = yt[train_idx], yt[val_idx]

            if task_type == "regression":
                w = _ridge_solve(Xtr_s, ytr, alpha=1.0)
                y_pred = Xval_s @ w
                yval_f = yval.to(DTYPE)
                ss_res = ((yval_f - y_pred) ** 2).sum()
                ss_tot = ((yval_f - yval_f.mean()) ** 2).sum()
                fold_scores.append(float(1.0 - ss_res / (ss_tot + 1e-12)))
            else:
                ytr_oh = torch.zeros(len(ytr), n_classes, dtype=DTYPE, device=device)
                ytr_oh.scatter_(1, ytr.unsqueeze(1), 1.0)
                W = _logistic_prox_grad(Xtr_s, ytr_oh, alpha_l1=0.0, alpha_l2=1e-4, max_iter=300)
                preds = (Xval_s @ W.T).argmax(dim=1)
                fold_scores.append(float((preds == yval).to(DTYPE).mean()))

        scores.append(float(np.mean(fold_scores)))

    return fractions, scores

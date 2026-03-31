"""
Feature selection methods for earth embedding analysis.

Methods
-------
fit_sparse_regression       — LassoCV + ElasticNetCV coefficients
fit_sparse_classification   — L1 / ElasticNet logistic regression coefficients
stability_selection         — bootstrap LASSO selection frequencies
compute_mi                  — mutual information per dimension
mrmr                        — greedy min-Redundancy Max-Relevance selection
progressive_ablation        — performance vs. fraction of top dims removed
"""

import numpy as np
from sklearn.linear_model import (
    LassoCV, ElasticNetCV, Lasso,
    LogisticRegression, Ridge,
)
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# LASSO & Elastic Net
# ---------------------------------------------------------------------------

def fit_sparse_regression(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit LassoCV and ElasticNetCV.

    Returns:
        (lasso_coef, enet_coef) — coefficients, shape (n_dims,)
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    X = StandardScaler().fit_transform(X)

    lasso = LassoCV(max_iter=5000, cv=3).fit(X, y)
    enet  = ElasticNetCV(max_iter=5000, cv=3).fit(X, y)
    return lasso.coef_, enet.coef_


def fit_sparse_classification(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit L1 and Elastic Net logistic regression.

    Returns:
        (l1_imp, enet_imp) — coefficients across classes
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], LabelEncoder().fit_transform(y[mask].astype(int))
    X = StandardScaler().fit_transform(X)

    l1   = LogisticRegression(penalty="l1",        solver="saga", max_iter=1000, C=0.1).fit(X, y)
    enet = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=1000, C=0.1, l1_ratio=0.5).fit(X, y)
    return l1.coef_, enet.coef_


# ---------------------------------------------------------------------------
# Stability Selection
# ---------------------------------------------------------------------------

def stability_selection(X: np.ndarray, y: np.ndarray, task_type: str,
                         n_bootstrap: int = 50, subsample_frac: float = 0.5,
                         alpha_frac: float = 0.1) -> np.ndarray:
    """Bootstrap subsampled LASSO to estimate selection frequency per dimension.

    Returns:
        selection_freq — shape (n_dims,), values in [0, 1]
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if task_type == "classification":
        y = LabelEncoder().fit_transform(y.astype(int))

    n = len(y)
    n_sub = int(n * subsample_frac)
    counts = np.zeros(X.shape[1])

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n_sub, replace=False) # needs to be seeded
        Xs, ys = X[idx], y[idx]
        Xs = StandardScaler().fit_transform(Xs)

        if task_type == "regression":
            alpha = np.std(ys) * alpha_frac
            m = Lasso(alpha=alpha, max_iter=5000)
        else:
            m = LogisticRegression(penalty="l1", C=1 / alpha_frac, solver="saga", max_iter=1000)
        m.fit(Xs, ys)

        coef = m.coef_ if task_type == "regression" else np.abs(m.coef_).mean(axis=0)
        counts += (np.abs(coef) > 1e-6).astype(float)

    return counts / n_bootstrap


# ---------------------------------------------------------------------------
# Mutual Information
# ---------------------------------------------------------------------------

def compute_mi(X: np.ndarray, y: np.ndarray, task_type: str,
               n_neighbors: int = 3) -> np.ndarray:
    """Compute mutual information between each dimension and the target.

    Returns:
        mi — shape (n_dims,)
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    X = StandardScaler().fit_transform(X)

    if task_type == "regression":
        return mutual_info_regression(X, y, n_neighbors=n_neighbors)
    else:
        return mutual_info_classif(X, y.astype(int), n_neighbors=n_neighbors)


# ---------------------------------------------------------------------------
# mRMR
# ---------------------------------------------------------------------------

def mrmr(X: np.ndarray, y: np.ndarray, task_type: str,
         n_features: int = 32, n_neighbors: int = 3) -> list[int]:
    """Greedy min-Redundancy Max-Relevance feature selection.

    Returns:
        selected — list of dim indices in selection order
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    X = StandardScaler().fit_transform(X)
    mi_fn = mutual_info_regression if task_type == "regression" else mutual_info_classif

    relevance = mi_fn(X, y.astype(int) if task_type == "classification" else y, n_neighbors=n_neighbors)
    selected = []
    remaining = list(range(X.shape[1]))

    for _ in range(n_features):
        if not remaining:
            break
        if not selected:
            best = remaining[int(np.argmax(relevance[remaining]))]
        else:
            redundancy = np.array([
                np.mean([mutual_info_regression(X[:, [s]], X[:, f])[0] for s in selected])
                for f in remaining
            ])
            best = remaining[int(np.argmax(relevance[remaining] - redundancy))]
        selected.append(best)
        remaining.remove(best)

    return selected


# ---------------------------------------------------------------------------
# Progressive Ablation
# ---------------------------------------------------------------------------

def progressive_ablation(X: np.ndarray, y: np.ndarray, task_type: str,
                          importance_scores: np.ndarray,
                          steps: int = 10, cv: int = 3) -> tuple[np.ndarray, list[float]]:
    """Progressively remove the most-important dims and track performance.

    Returns:
        (fractions, scores) — fractions removed (0..0.9), CV score at each step
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    ranked_dims = np.argsort(importance_scores)[::-1]
    n_dims = X.shape[1]
    fractions = np.linspace(0, 0.9, steps)
    scores = []

    for frac in fractions:
        n_remove = int(frac * n_dims)
        keep = ranked_dims[n_remove:]
        Xk = StandardScaler().fit_transform(X[:, keep])
        if task_type == "regression":
            score = cross_val_score(Ridge(), Xk, y, cv=cv, scoring="r2").mean()
        else:
            score = cross_val_score(
                LogisticRegression(max_iter=500), Xk, y.astype(int), cv=cv, scoring="accuracy"
            ).mean()
        scores.append(score)

    return fractions, scores

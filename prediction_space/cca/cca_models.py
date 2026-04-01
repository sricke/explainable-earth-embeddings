"""
CCA-based interpretability for earth embeddings.

For each (embedding, task) pair, fits a CCA model and returns:
  - canonical weights on the X side   (n_dims, n_components)
  - canonical loadings on the X side  (n_dims, n_components)  ← correlation of each dim with the canonical variate
  - canonical correlations             (n_components,)

Weights tell you how to form the canonical variate; loadings tell you how correlated
each original dimension is with that variate — more interpretable for "which dims
matter for elevation?".

Also provides compute_dim_label_correlations for the transpose view:
"for embedding dimension k, what combination of environmental variables correlates with it?"
"""

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def _prepare_Y(y, task_type: str):
    """Return a 2-D float array for CCA's Y side."""
    if task_type == "regression":
        return y.reshape(-1, 1).astype(float)
    else:
        classes = np.unique(y[~np.isnan(y)]).astype(int)
        return (y[:, None] == classes[None, :]).astype(float)


def compute_cca(X: np.ndarray, y: np.ndarray, task_type: str,
                n_components: int = 1):
    """
    Fit CCA between embedding X and target y.

    Returns
    -------
    weights : np.ndarray, shape (n_dims, n_comp)
        Canonical weight vectors on the X side (used to form canonical variates).
    loadings : np.ndarray, shape (n_dims, n_comp)
        Pearson r between each original X dimension and each X canonical variate.
        More interpretable than weights: tells you which dims are associated with
        the canonical direction, without sign ambiguity from collinearity.
    corrs : np.ndarray, shape (n_comp,)
        Pearson r between paired X and Y canonical variates.
    n_comp : int
        Actual number of components fit (may be < n_components for regression
        tasks where Y is 1-D, capping at 1).
    """
    Y = _prepare_Y(y, task_type)

    # Drop rows with any NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X_c, Y_c = X[mask], Y[mask]

    # Standardise
    X_s = StandardScaler().fit_transform(X_c)
    Y_s = StandardScaler().fit_transform(Y_c)

    n_comp = min(n_components, X_s.shape[1], Y_s.shape[1])
    model = CCA(n_components=n_comp, max_iter=1000)
    model.fit(X_s, Y_s)

    Xt, Yt = model.transform(X_s, Y_s)
    corrs = np.array([np.corrcoef(Xt[:, i], Yt[:, i])[0, 1] for i in range(n_comp)])

    # Loadings: correlation of each original X dim with each canonical variate
    loadings = np.array([
        [np.corrcoef(X_s[:, d], Xt[:, i])[0, 1] for i in range(n_comp)]
        for d in range(X_s.shape[1])
    ])  # (n_dims, n_comp)

    return model.x_weights_, loadings, corrs, n_comp


def compute_cca_vars_to_dim(
    env_vars_matrix: np.ndarray,  # (n_samples, n_env_vars)
    dim_values: np.ndarray,       # (n_samples,)
) -> tuple:
    """
    CCA with X = all env variables jointly, Y = single embedding dimension.

    Finds the linear combination (projection) of env vars most correlated with
    that embedding dimension.

    Returns
    -------
    weights : np.ndarray, shape (n_env_vars,)
        CCA weight vector on the env-vars side (the projection direction).
    corr : float
        Canonical correlation between the env-var projection and the embedding dim.
    """
    Y = dim_values.reshape(-1, 1).astype(float)
    mask = ~(np.isnan(env_vars_matrix).any(axis=1) | np.isnan(Y).any(axis=1))
    X_c, Y_c = env_vars_matrix[mask], Y[mask]

    if X_c.shape[0] < 10:
        return np.zeros(env_vars_matrix.shape[1]), 0.0

    X_s = StandardScaler().fit_transform(X_c)
    Y_s = StandardScaler().fit_transform(Y_c)

    model = CCA(n_components=1, max_iter=1000)
    model.fit(X_s, Y_s)

    Xt, Yt = model.transform(X_s, Y_s)
    corr = float(np.corrcoef(Xt[:, 0], Yt[:, 0])[0, 1])

    return model.x_weights_[:, 0], corr  # (n_env_vars,), float


def compute_dim_label_correlations(
    embeddings_by_dataset: dict[str, np.ndarray],
    all_labels: dict,
    task_names: list,
) -> np.ndarray:
    """
    Compute Pearson r between each embedding dimension and each regression label.

    Parameters
    ----------
    embeddings_by_dataset
        For one embedding model: ``dataset_name -> X`` with ``X`` shape ``(n_ds, n_dims)`` for that task.
    all_labels : dict mapping task_name -> (y, task_type)
    task_names : ordered list of task names to use as columns

    Returns
    -------
    corr_mat : np.ndarray, shape (n_dims, n_tasks)
        Entry [d, t] = Pearson r between X[:,d] and label t.
        NaN for classification tasks (labels are nominal, not ordinal).
    """
    n_dims = next(iter(embeddings_by_dataset.values())).shape[1]
    n_tasks = len(task_names)
    corr_mat = np.full((n_dims, n_tasks), np.nan)

    for t_idx, ds_name in enumerate(task_names):
        y, task_type = all_labels[ds_name]
        if task_type != "regression":
            continue
        X = embeddings_by_dataset[ds_name]
        valid = ~np.isnan(y)
        y_v = y[valid].astype(float)
        X_v = X[valid]
        # Vectorised: corr(X_v[:,d], y_v) for all d at once
        X_centered = X_v - X_v.mean(axis=0, keepdims=True)
        y_centered = y_v - y_v.mean()
        num   = X_centered.T @ y_centered                           # (n_dims,)
        denom = np.sqrt((X_centered ** 2).sum(axis=0)) * np.sqrt((y_centered ** 2).sum())
        with np.errstate(invalid="ignore"):
            corr_mat[:, t_idx] = np.where(denom > 0, num / denom, np.nan)

    return corr_mat

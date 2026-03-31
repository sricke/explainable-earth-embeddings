"""
sklearn kNN probes for regression and classification.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold

K_VALUES = [1, 3, 5, 10]


def probe_knn_regression(X: np.ndarray, y: np.ndarray, k_values: list = K_VALUES,
                         cv: int = 5, seed: int = 42) -> dict[int, tuple[float, float]]:
    """kNN regression probe.

    Returns:
        dict mapping k -> (mean_r2, std_r2)
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    results = {}
    for k in k_values:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=k))])
        scores = cross_val_score(pipe, X, y, cv=splitter, scoring="r2")
        results[k] = (scores.mean(), scores.std())
    return results


def probe_knn_classification(X: np.ndarray, y: np.ndarray, k_values: list = K_VALUES,
                              cv: int = 5, seed: int = 42) -> dict[int, tuple[float, float]]:
    """kNN classification probe.

    Returns:
        dict mapping k -> (mean_accuracy, std_accuracy)
    """
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], LabelEncoder().fit_transform(y[mask].astype(int))

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    results = {}
    for k in k_values:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=k))])
        scores = cross_val_score(pipe, X, y, cv=splitter, scoring="accuracy")
        results[k] = (scores.mean(), scores.std())
    return results

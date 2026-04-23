import numpy as np

from .dataset import REGRESSION_TASKS


def top_concepts(task: str, activations: np.ndarray, y_train: np.ndarray, k: int):
    """
    Regression: returns array of shape (k,) — top-k concept indices by mean activation.
    Classification: returns dict {class_label: array(k,)} — per-class top-k indices.
    """
    if task in REGRESSION_TASKS:
        return np.argsort(activations.mean(axis=0))[::-1][:k]

    classes = np.unique(y_train.astype(int))
    return {
        c: np.argsort(activations[y_train.astype(int) == c].mean(axis=0))[::-1][:k]
        for c in classes
    }

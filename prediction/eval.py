import numpy as np
from sklearn.metrics import r2_score, accuracy_score

from .dataset import REGRESSION_TASKS

def eval_ridge(task: str, model, x: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(x)
    if task in REGRESSION_TASKS:
        return r2_score(y, y_pred)
    return accuracy_score(y.astype(int), y_pred)

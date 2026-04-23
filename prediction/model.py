from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.model_selection import GridSearchCV

from .dataset import CLASSIFICATION_TASKS


def fit_ridge(task: str, x_train, y_train, cv: bool = True):
    """Fit and return a Ridge probe (regression or classifier) on sparse codes."""
    cls = RidgeClassifier if task in CLASSIFICATION_TASKS else Ridge
    if cv:
        grid = GridSearchCV(cls(), {"alpha": [0.01, 0.1, 1, 10, 100]}, cv=5)
        grid.fit(x_train, y_train)
        print(f"[{task}] best alpha: {grid.best_params_['alpha']}")
        return grid.best_estimator_
    model = cls(alpha=1.0)
    model.fit(x_train, y_train)
    return model

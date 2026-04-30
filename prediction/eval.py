import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
import shap
from lime.lime_tabular import LimeTabularExplainer


def top_concepts(model: Ridge | RidgeCV, k: int = 10, class_labels=None):
    weights = np.asarray(model.coef_)
    topk = lambda w: np.argsort(np.abs(w))[::-1][:k]
    if weights.ndim == 1:
        return topk(weights)
    labels = class_labels if class_labels is not None else range(len(weights))
    return {cls: topk(row) for cls, row in zip(labels, weights)}


def shap_explanations(model, X):
    """Returns SHAP values array (n_samples, n_features)."""
    explainer = shap.LinearExplainer(model, X)
    return explainer.shap_values(X)


def lime_explanations(model, X_train, X_explain, n_samples: int = 1000):
    """Returns list of LIME explanation objects, one per row of X_explain."""
    explainer = LimeTabularExplainer(X_train, mode="regression")
    return [
        explainer.explain_instance(row, model.predict, num_samples=n_samples)
        for row in X_explain
    ]

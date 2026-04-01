import sys
import warnings
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.append("..")


def probe_ridge(
    X_train,
    y_train,
    X_test,
    y_test,
    alphas=None,
    cv=5,
    seed=42,
    solver="cholesky",
):
    """
    Pick ``alpha`` by K-fold CV R² on the training set, refit on all training rows, return test R².
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10, 100]

    m_tr = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    m_te = np.isfinite(X_test).all(axis=1) & np.isfinite(y_test)
    X_tr, y_tr = X_train[m_tr], y_train[m_tr]
    X_te, y_te = X_test[m_te], y_test[m_te]
    if X_tr.shape[0] < cv:
        raise ValueError(
            f"Training rows after filtering ({X_tr.shape[0]}) must be >= cv folds ({cv})"
        )
    if X_te.shape[0] < 1:
        raise ValueError("Need at least one finite test row for evaluation")

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    mean_cv_r2: dict[float, float] = {}
    std_cv_r2: dict[float, float] = {}

    for alpha in alphas:
        fold_scores = []
        for tr_idx, va_idx in splitter.split(X_tr):
            Xa, Xb = X_tr[tr_idx], X_tr[va_idx]
            ya, yb = y_tr[tr_idx], y_tr[va_idx]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=alpha, solver=solver)),
                ]
            )
            pipe.fit(Xa, ya)
            fold_scores.append(pipe.score(Xb, yb))
        mean_cv_r2[float(alpha)] = float(np.mean(fold_scores))
        std_cv_r2[float(alpha)] = float(np.std(fold_scores))

    best_alpha = max(mean_cv_r2, key=mean_cv_r2.get)
    best_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=best_alpha, solver=solver)),
        ]
    )
    best_pipe.fit(X_tr, y_tr)
    test_r2 = float(best_pipe.score(X_te, y_te))

    alpha_info = {
        "mean_cv_r2": mean_cv_r2,
        "std_cv_r2": std_cv_r2,
        "best_alpha": best_alpha,
        "selection_counts": {a: cv if a == best_alpha else 0 for a in mean_cv_r2},
    }
    return test_r2, std_cv_r2[best_alpha], alpha_info


def probe_logistic(X, y, Cs=None, cv=5, seed=42, solver='lbfgs'):
    if Cs is None:
        Cs = [0.001, 0.01, 0.1, 1.0, 10, 100]  # inverse regularization

    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = LabelEncoder().fit_transform(y[mask].astype(int))

    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)

    best_C = None
    best_score = -np.inf
    best_std = 0.0

    for C in Cs:
        cv_scores = []

        for train_idx, val_idx in splitter.split(X):
            Xtr, Xva = X[train_idx], X[val_idx]
            ytr, yva = y[train_idx], y[val_idx]

            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    multi_class='multinomial',
                    random_state=seed
                ))
            ])

            pipe.fit(Xtr, ytr)
            score = pipe.score(Xva, yva)
            cv_scores.append(score)

        mean_score = float(np.mean(cv_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_std = float(np.std(cv_scores))
            best_C = C

    return best_score, best_std

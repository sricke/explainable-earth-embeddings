import sys
import warnings
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import cross_val_score, KFold

warnings.filterwarnings("ignore")

sys.path.append("..")


def probe_ridge(X, y, cv=5, seed=42):
    mask = np.isfinite(X).all(axis=1) & ~np.isnan(X).any(axis=1) & np.isfinite(y) & ~np.isnan(y) # is finite and not NaN
    X, y = X[mask], y[mask]
    
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RidgeCV())
    ])

    score = cross_val_score(pipe, X, y, cv=splitter, scoring='r2')
    
    return score.mean(), score.std()

def probe_logistic(X, y, cv=5, seed=42):
    mask = np.isfinite(X).all(axis=1) & ~np.isnan(X).any(axis=1) & np.isfinite(y) & ~np.isnan(y) # is finite and not NaN
    X, y = X[mask], LabelEncoder().fit_transform(y[mask].astype(int))
    
    splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegressionCV(
            max_iter=1000,
            multi_class='multinomial',
            random_state=seed  # controls internal randomness
        ))
    ])

    score = cross_val_score(pipe, X, y, cv=splitter, scoring='accuracy')
    
    return score.mean(), score.std()
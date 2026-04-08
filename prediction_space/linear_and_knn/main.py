import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import faiss
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.utils.data import TensorDataset, random_split

sys.path.append("..")
from prediction_datasets.datasets import load_dataset
from utils import plot_linear_results, plot_linear_heatmap, plot_knn_results, plot_knn_heatmap

REGRESSION_TASKS = ['air_temp', 'elevation', 'nightlights', 'pop_density', 'treecover']
CLASSIFICATION_TASKS = ['biome', 'ecoregion']#, 'countries']
BACKENDS = ['geoclip', 'satclip', 'aef']


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def run_ridge_cv(x, y, model_cls=Ridge):
    param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(model_cls(), param_grid, cv=5)
    grid_search.fit(x, y)
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best alpha: {best_alpha}")
    return best_alpha


def run_single_ridge_regression(x_train, y_train, x_test, y_test, alpha=None, run_cv=True):
    if not run_cv and alpha is None:
        raise ValueError("Must provide alpha if not running CV")
    if run_cv:
        alpha = run_ridge_cv(x_train, y_train)

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2: {r2:.4f}")
    return r2


def run_single_ridge_classifier(x_train, y_train, x_test, y_test, alpha=None, run_cv=True):
    if not run_cv and alpha is None:
        raise ValueError("Must provide alpha if not running CV")
    if run_cv:
        alpha = run_ridge_cv(x_train, y_train, model_cls=RidgeClassifier)

    model = RidgeClassifier(alpha=alpha)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return acc

def run_knn_regression_fast(x_train, y_train, x_test, y_test, k=5):
    #tensor to numpy float32 for faiss
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test, dtype=np.float32)

    index = faiss.IndexFlatL2(x_train.shape[1])
    index.add(x_train)

    distances, indices = index.search(x_test, k)
    y_pred = y_train[indices].mean(axis=1)

    r2 = r2_score(y_test, y_pred)
    print(f"KNN Regression R2: {r2:.4f}")
    return r2

def run_knn_classification_fast(x_train, y_train, x_test, y_test, k=5):
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test  = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)

    index = faiss.IndexFlatL2(x_train.shape[1])
    index.add(x_train)

    distances, indices = index.search(x_test, k)  # (n_test, k)
    neighbor_labels = y_train[indices]            # (n_test, k)

    # Majority vote per row
    y_pred = np.array([
        np.bincount(row).argmax()
        for row in neighbor_labels
    ])

    acc = accuracy_score(y_test, y_pred)
    print(f"KNN Classification Accuracy: {acc:.4f}")
    return acc


# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------

def split_dataset(lat, lon, x, y, seed=42):
    coords = torch.tensor(np.column_stack([lon, lat]))
    x = torch.tensor(x)
    y = torch.tensor(y)
    dataset = TensorDataset(coords, x, y)
    n = len(dataset)
    train_size = int(0.5 * n)
    train_set, test_set = random_split(
        dataset, [train_size, n - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    def _unpack(split):
        t = split.dataset.tensors
        idx = split.indices
        return t[0][idx], t[1][idx], t[2][idx]
    return _unpack(train_set), _unpack(test_set)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rerun even if results CSV exists")
    parser.add_argument("--knn-subsample", type=int, default=None, metavar="N",
                        help="Subsample N training points for KNN (default: use all)")
    args = parser.parse_args()

    regression_csv_path = Path("regression_results.csv")

    if regression_csv_path.exists() and not args.force:
        print(f"Loading results from {regression_csv_path}")
        df = pd.read_csv(regression_csv_path)
        print(df.to_string(index=False))
    else:
        records = []

        for task in CLASSIFICATION_TASKS:
            for backend in BACKENDS:
                print(f"Logistic regression — {task} / {backend}")
                ds = load_dataset(task, backend, device='cpu')
                lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']
                (_, x_train, y_train), (_, x_test, y_test) = split_dataset(lat, lon, x, y.astype(int))
                y_train, y_test = y_train.long(), y_test.long()
                acc = run_single_ridge_classifier(x_train, y_train, x_test, y_test)
                records.append({"task": task, "backend": backend, "metric": "accuracy", "value": acc})

        for task in REGRESSION_TASKS:
            for backend in BACKENDS:
                print(f"Ridge regression — {task} / {backend}")
                ds = load_dataset(task, backend, device='cpu')
                lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']

                (_, x_train, y_train), (_, x_test, y_test) = split_dataset(lat, lon, x, y)
                max_norm_val = y_train.max().item()
                y_train = y_train / max_norm_val
                y_test = y_test / max_norm_val
                r2 = run_single_ridge_regression(x_train, y_train, x_test, y_test, alpha=0.1, run_cv=False)
                records.append({"task": task, "backend": backend, "metric": "r2", "value": r2})

        df = pd.DataFrame(records)
        df.to_csv(regression_csv_path, index=False)
        print(df.to_string(index=False))

    out_dir = Path("figures")
    plot_linear_results(df, out_path=out_dir / "linear_results.png")
    plot_linear_heatmap(df, out_path=out_dir / "linear_heatmap.png")

    knn_csv_path = Path("knn_results.csv")
    if knn_csv_path.exists() and not args.force:
        print(f"Loading KNN results from {knn_csv_path}")
        knn_df = pd.read_csv(knn_csv_path)
        print(knn_df.to_string(index=False))
    else:
        knn_records = []
        rng = np.random.default_rng(42)

        for task in REGRESSION_TASKS:
            for backend in BACKENDS:
                for k in [1, 3, 5, 10]:
                    print(f"KNN regression — {task} / {backend} (k={k})")
                    ds = load_dataset(task, backend, device='cpu')
                    lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']
                    (_, x_train, y_train), (_, x_test, y_test) = split_dataset(lat, lon, x, y)

                    max_norm_val = y_train.max().item()
                    y_train = y_train / max_norm_val
                    y_test = y_test / max_norm_val

                    n_train = len(x_train)
                    if args.knn_subsample is not None and args.knn_subsample < n_train:
                        #seeded
                        rng = np.random.default_rng(42)
                        idx = rng.choice(n_train, args.knn_subsample, replace=False)
                        x_train_knn = x_train[idx]
                        y_train_knn = np.asarray(y_train)[idx]
                    else:
                        x_train_knn, y_train_knn = x_train, y_train

                    r2 = run_knn_regression_fast(x_train_knn, y_train_knn, x_test, y_test, k=k)
                    knn_records.append({"task": task, "backend": backend, "k": k,
                                        "subsample": args.knn_subsample if args.knn_subsample is not None else n_train,
                                        "metric": "r2", "value": r2})

        for task in CLASSIFICATION_TASKS:
            for backend in BACKENDS:
                for k in [1, 3, 5, 10]:
                    print(f"KNN classification — {task} / {backend} (k={k})")
                    ds = load_dataset(task, backend, device='cpu')
                    lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']
                    (_, x_train, y_train), (_, x_test, y_test) = split_dataset(lat, lon, x, y.astype(int))
                    y_train, y_test = y_train.long(), y_test.long()

                    n_train = len(x_train)
                    if args.knn_subsample is not None and args.knn_subsample < n_train:
                        rng = np.random.default_rng(42)
                        idx = rng.choice(n_train, args.knn_subsample, replace=False)
                        x_train_knn = x_train[idx]
                        y_train_knn = np.asarray(y_train)[idx]
                    else:
                        x_train_knn, y_train_knn = x_train, y_train

                    acc = run_knn_classification_fast(x_train_knn, y_train_knn, x_test, y_test, k=k)
                    knn_records.append({"task": task, "backend": backend, "k": k,
                                        "subsample": args.knn_subsample if args.knn_subsample is not None else n_train,
                                        "metric": "accuracy", "value": acc})

        knn_df = pd.DataFrame(knn_records)
        knn_df.to_csv("knn_results.csv", index=False)
        print(knn_df.to_string(index=False))

    plot_knn_results(knn_df, out_path=out_dir / "knn_results.png")
    plot_knn_heatmap(knn_df, out_path=out_dir / "knn_heatmap.png")

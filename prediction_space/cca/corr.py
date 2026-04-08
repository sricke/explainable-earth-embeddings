"""
Scalar-Y interpretability via maximal linear association with embeddings.

With a single column of Y, sklearn's CCA(n_components=1) is the standard
maximal-correlation construction: the canonical correlation equals the
multiple correlation R between Y and X (after CCA's internal centering /
scaling). The X-side direction matches the standardized linear regression
direction (same 1-D subspace as OLS on scaled predictors). Multivariate CCA
can be added in a separate script when Y has several columns.
"""
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from torch.utils.data import TensorDataset, random_split

sys.path.append("..")
from prediction_datasets.datasets import load_dataset
from utils import (
    plot_cca_weight_heatmap,
    plot_cca_weight_heatmap_clustered,
    plot_variable_loadings,
    plot_canonical_correlations,
)

REGRESSION_TASKS = ['air_temp', 'elevation', 'nightlights', 'pop_density', 'treecover']
CLASSIFICATION_TASKS = ['biome', 'ecoregion']
ALL_TASKS = REGRESSION_TASKS + CLASSIFICATION_TASKS
BACKENDS = ['geoclip', 'satclip', 'aef']
N_COMPONENTS = 1   # scalar Y → one linear score; R = multiple correlation
METHOD_LABEL = "Linear regression (scalar Y)"
DATA_OUT = Path("figures/corr/data")
FIG_OUT = Path("figures/corr")
# Primary cache name; legacy `cca_{backend}.npz` is still loaded if present.
NPZ_STEM = "scalar_y_linear"


# ---------------------------------------------------------------------------
# Scalar-Y linear association (implemented with sklearn CCA for one component)
# ---------------------------------------------------------------------------

def scalar_y_linear_weights(x_train, y_train, x_test, y_test, n_components=10):
    cca = CCA(n_components=n_components)
    cca.fit(x_train.float().cpu(), y_train.float().cpu())
    x_train_cca, y_train_cca = cca.transform(x_train.float().cpu(), y_train.float().cpu())
    x_test_cca, y_test_cca = cca.transform(x_test.float().cpu(), y_test.float().cpu())

    r2 = r2_score(y_test_cca, cca.inverse_transform(x_test_cca))
    print(f"Linear score R² (holdout reconstruction): {r2:.4f}")

    return cca.x_weights_, cca.y_weights_


def split_dataset(lat, lon, x, y, seed=42):
    coords = torch.tensor(np.column_stack([lon, lat]))
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    y = y if isinstance(y, torch.Tensor) else torch.tensor(y)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    x_weights_by_backend = {b: {} for b in BACKENDS}  # [backend][task] = (n_dims, 1)
    multiple_r_by_backend = {b: {} for b in BACKENDS}  # [backend][task] = R (scalar Y)

    for backend in BACKENDS:
        npz_path = DATA_OUT / f"{NPZ_STEM}_{backend}.npz"
        legacy_path = DATA_OUT / f"cca_{backend}.npz"
        load_path = npz_path if npz_path.exists() else legacy_path

        if load_path.exists() and not args.force:
            d = np.load(load_path, allow_pickle=True)
            for task in ALL_TASKS:
                if f"xw_{task}" in d:
                    x_weights_by_backend[backend][task] = d[f"xw_{task}"]
                    multiple_r_by_backend[backend][task] = d[f"corr_{task}"]
            print(f"Loaded {load_path}")
            continue

        print(f"\n{'='*50}\nBackend: {backend}\n{'='*50}")
        save_dict = {}

        for task in ALL_TASKS:
            print(f"  {METHOD_LABEL} — {task}")
            ds = load_dataset(task, backend, device='cpu')
            lat, lon, y, x = ds['lat'], ds['lon'], ds['value'], ds['embeddings']

            x_np = np.asarray(x, dtype=np.float32)
            y_np = y.astype(np.float32)

            (_, x_tr, y_tr), _ = split_dataset(lat, lon, x_np, y_np)
            x_tr = np.asarray(x_tr, dtype=np.float32)
            y_tr = np.asarray(y_tr, dtype=np.float32).reshape(-1, 1)

            cca = CCA(n_components=N_COMPONENTS)
            cca.fit(x_tr, y_tr)
            x_t, y_t = cca.transform(x_tr, y_tr)
            r = float(np.corrcoef(x_t[:, 0], y_t[:, 0])[0, 1])
            print(f"    multiple correlation R = {r:.4f}")

            x_weights_by_backend[backend][task] = cca.x_weights_
            multiple_r_by_backend[backend][task] = np.array([r])
            save_dict[f"xw_{task}"] = cca.x_weights_
            save_dict[f"corr_{task}"] = np.array([r])

        np.savez(npz_path, **save_dict)
        print(f"Saved {npz_path}")

    # ---- Plots ---------------------------------------------------------------

    plot_canonical_correlations(
        multiple_r_by_backend, ALL_TASKS, n_components=N_COMPONENTS,
        out_path=FIG_OUT / "multiple_correlation_scalar_y.png",
        method_label=METHOD_LABEL,
        score_label="Multiple correlation R",
    )

    plot_cca_weight_heatmap(
        x_weights_by_backend, ALL_TASKS, component=0,
        out_path=FIG_OUT / "linear_score_loadings_heatmap.png",
        method_label=METHOD_LABEL,
        loading_bar_label="Linear score loading",
    )

    plot_cca_weight_heatmap_clustered(
        x_weights_by_backend, ALL_TASKS, component=0,
        out_path=FIG_OUT / "linear_score_loadings_heatmap.png",
        method_label=METHOD_LABEL,
        loading_bar_label="Linear score loading",
    )

    plot_variable_loadings(
        x_weights_by_backend, ALL_TASKS, top_k=30, component=0,
        out_path=FIG_OUT / "variable_loadings.png",
        method_label=METHOD_LABEL,
        projection_xlabel="Linear score weight",
    )

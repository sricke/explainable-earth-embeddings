"""
Multi-seed evaluation of SpLiCE reconstruction quality.

For each model, samples SUBSET_SIZE locations N_SEEDS times (different random seeds),
fits a fresh SPLICE mean embedding per seed (fixed dictionary), and reports
mean ± std of MSE and avg cosine similarity across seeds.
"""

import sys
sys.path.append('..')

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from splice import _LocWrapper  # needed for unpickling saved SPLICE models
from external.splice.splice.admm import ADMM

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DENSE_GRID_CSV = "/home/libe2152/data/dense_grid/dense_grid.csv"
BATCH_SIZE = 2048
N_SEEDS = 3
SUBSET_SIZE = 10_000

CONFIGS = {
    "satclip_open_clip_vit_l": {
        "splice_model": "../splice_results/satclip_open_clip_vit_l/geospatial/splice_model.pt",
        "l1_penalty": 0.175,
    },
    "geoclip": {
        "splice_model": "../splice_results/geoclip/geospatial/splice_model.pt",
        "l1_penalty": 0.175,
    },
    "csp_fmow_open_clip_vit_l": {
        "splice_model": "../splice_results/csp_fmow_open_clip_vit_l/geospatial/splice_model.pt",
        "l1_penalty": 0.125,
    },
    "climplicit_open_clip_vit_l": {
        "splice_model": "../splice_results/climplicit_open_clip_vit_l/geospatial/splice_model.pt",
        "l1_penalty": 0.125,
    },
}


def recompose(weights, dictionary, image_mean):
    recon = weights @ dictionary
    recon = F.normalize(recon, dim=1)
    recon = F.normalize(recon + image_mean, dim=1)
    return recon


def run_one_seed(loc_embs, admm, dictionary, device):
    """
    loc_embs: (N, D) raw (pre-norm) location embeddings for this seed's subsample.
    Returns (mse, avg_cosine, avg_active_concepts).
    """
    orig = F.normalize(loc_embs, dim=1)            # L2-normalize
    image_mean = orig.mean(0)                       # seed-specific mean

    all_orig, all_recon, all_weights = [], [], []
    for i in range(0, len(orig), BATCH_SIZE):
        orig_b = orig[i : i + BATCH_SIZE]
        centered_b = F.normalize(orig_b - image_mean, dim=1)
        with torch.no_grad():
            weights_b = admm.fit(centered_b).to(device)
        recon_b = recompose(weights_b, dictionary, image_mean)
        all_orig.append(orig_b)
        all_recon.append(recon_b)
        all_weights.append(weights_b)

    orig_all = torch.cat(all_orig)
    recon_all = torch.cat(all_recon)
    weights_all = torch.cat(all_weights)

    mse = F.mse_loss(recon_all, orig_all).item()
    avg_cos = (recon_all * orig_all).sum(dim=1).mean().item()
    avg_active = (weights_all > 0).float().sum(dim=1).mean().item()
    return mse, avg_cos, avg_active


def evaluate_model(model_name, cfg, df_full):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  L1={cfg['l1_penalty']}")
    print(f"{'='*60}")

    splice = torch.load(cfg["splice_model"], map_location=DEVICE, weights_only=False)
    splice.eval()
    pca = getattr(splice, "pca", None)

    # dictionary is already row-normalized inside SPLICE.__init__
    dictionary = splice.dictionary.to(DEVICE)

    # Build one ADMM solver for this model (set_C does the Cholesky decomp — expensive)
    admm = ADMM(
        rho=5.0,
        l1_penalty=cfg["l1_penalty"],
        tol=1e-7,
        max_iter=2000,
        device=DEVICE,
        verbose=False,
    )
    admm.set_C(dictionary)

    mses, cosines, actives = [], [], []

    for seed in tqdm(range(N_SEEDS), desc=f"Seeds for {model_name}"):
        sample_df = df_full.sample(n=SUBSET_SIZE, random_state=seed)
        latlons = torch.tensor(
            sample_df[["lat", "lon"]].values, dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():
            raw = splice.clip.encode_image(latlons)

        if pca is not None:
            raw = torch.tensor(
                pca.transform(raw.cpu().numpy()), dtype=torch.float32
            ).to(DEVICE)

        mse, avg_cos, avg_active = run_one_seed(raw, admm, dictionary, DEVICE)
        mses.append(mse)
        cosines.append(avg_cos)
        actives.append(avg_active)
        print(f"  seed {seed:2d}: MSE={mse:.5f}  cos={avg_cos:.4f}  active={avg_active:.1f}")

    result = {
        "model": model_name,
        "l1_penalty": cfg["l1_penalty"],
        "n_seeds": N_SEEDS,
        "subset_size": SUBSET_SIZE,
        "mse_mean": float(np.mean(mses)),
        "mse_std": float(np.std(mses, ddof=1)),
        "cosine_mean": float(np.mean(cosines)),
        "cosine_std": float(np.std(cosines, ddof=1)),
        "active_concepts_mean": float(np.mean(actives)),
        "active_concepts_std": float(np.std(actives, ddof=1)),
        "mse_per_seed": mses,
        "cosine_per_seed": cosines,
        "active_concepts_per_seed": actives,
    }

    print(f"\n  MSE:            {result['mse_mean']:.8f} ± {result['mse_std']:.8f}")
    print(f"  Cosine:         {result['cosine_mean']:.8f} ± {result['cosine_std']:.8f}")
    print(f"  Active concepts:{result['active_concepts_mean']:.2f} ± {result['active_concepts_std']:.2f}")
    return result


def main():
    df_full = pd.read_csv(DENSE_GRID_CSV)
    print(f"Loaded dense grid: {len(df_full)} locations")

    all_results = {}
    for model_name, cfg in CONFIGS.items():
        result = evaluate_model(model_name, cfg, df_full)
        all_results[model_name] = result

    out_path = Path("reports/eval_splice_seeds_geospatial.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")

    print("\n\n=== SUMMARY ===")
    print(f"{'Model':<35} {'L1':>6}  {'MSE (mean±std)':>24}  {'Cosine (mean±std)':>24}  {'Active (mean±std)':>20}")
    print("-" * 115)
    for name, r in all_results.items():
        print(
            f"{name:<35} {r['l1_penalty']:>6.3f}  "
            f"{r['mse_mean']:.8f} ± {r['mse_std']:.8f}  "
            f"{r['cosine_mean']:.8f} ± {r['cosine_std']:.8f}  "
            f"{r['active_concepts_mean']:.2f} ± {r['active_concepts_std']:.2f}"
        )


if __name__ == "__main__":
    main()

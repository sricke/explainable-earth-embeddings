"""
Diagnostic: why does satclip SpLiCE reconstruct perfectly while climplicit fails?

Existing reports tell us:
  satclip_geospatial_all:   r2=0.9999, cos=0.9999, avg_active=982   <- perfect
  satclip_test_2000:        r2=-0.75,  cos=0.18,   avg_active=4.7   <- bad
  climplicit_geospatial_all: r2=-10.5, cos=0.54,   avg_active=60    <- bad
  climplicit_test_2000:     r2=-12.6,  cos=0.20,   avg_active=4.5   <- bad

We use both geospatial_all models (500K concepts) as the comparison pair:
satclip CAN reach r2≈1 with ADMM; climplicit cannot, even with 500K concepts.
"""
import sys
sys.path.append("..")

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
from pathlib import Path

from splice import _LocWrapper  # noqa: F401
from paths import load_paths

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N      = 1000
LAT_LON_CSV = "/home/libe2152/data/dense_grid/dense_grid.csv"

_p = load_paths()

# Use geospatial_all (500K concepts) — this is the setting where satclip succeeds
MODELS = {
    "satclip":    _p["splice"]["satclip_geospatial_all"]["model"],
    "climplicit": _p["splice"]["climplicit_geospatial_all"]["model"],
}


# ── helpers ──────────────────────────────────────────────────────────────────

def recompose(w, C, mean):
    """Standard SpLiCE: normalize(normalize(w@C) + mean)"""
    r = F.normalize(w @ C, dim=1)
    return F.normalize(r + mean, dim=1)

def cos_sim(a, b):
    return (F.normalize(a, dim=1) * F.normalize(b, dim=1)).sum(dim=1)

def metrics(orig, recon, label=""):
    cos  = cos_sim(orig, recon).mean().item()
    mse  = ((orig - recon) ** 2).mean().item()
    r2   = r2_score(orig.cpu().numpy(), recon.cpu().numpy(), multioutput="uniform_average")
    print(f"  [{label:45s}]  cos={cos:.4f}  mse={mse:.7f}  r2={r2:+.4f}")
    return cos, mse, r2

def oracle_weights(centered, C):
    """Min-norm unconstrained solution: w @ C = centered (exact).
    For K >> D: w = centered @ (C^T C)^{-1} @ C^T  (shape N×K)
    Avoids computing the K×K matrix."""
    CtC = C.T @ C                                        # D×D
    return centered @ torch.linalg.pinv(CtC) @ C.T      # N×K


# ── main loop ─────────────────────────────────────────────────────────────────

df     = pd.read_csv(LAT_LON_CSV).sample(n=N, random_state=42)
latlons = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(DEVICE)

for name, model_path in MODELS.items():
    print(f"\n{'='*75}")
    print(f"  MODEL: {name}  ({model_path})")
    print(f"{'='*75}")

    splice = torch.load(model_path, map_location=DEVICE, weights_only=False)
    splice.eval()

    mean = splice.image_mean                   # (D,)
    C    = splice.dictionary.to(DEVICE)        # (K, D)
    K, D = C.shape
    print(f"  dict shape: {K} × {D},  image_mean norm: {mean.norm():.4f}")

    # ── Get location embeddings ───────────────────────────────────────────────
    with torch.no_grad():
        orig = F.normalize(splice.clip.encode_image(latlons), dim=1)   # (N, D)

    centered = F.normalize(orig - mean, dim=1)

    # ── 0. Embedding geometry ─────────────────────────────────────────────────
    print(f"\n--- 0. Embedding geometry ---")
    var_per_dim = orig.var(0).mean().item()
    print(f"  per-dim variance (orig)     : {var_per_dim:.6f}  (uniform-sphere ref: {1/D:.6f})")
    print(f"  mean embedding (norm)       : {orig.mean(0).norm():.4f}  "
          f"(high = concentrated cluster)")

    # ── 1. Loc ↔ concept alignment ────────────────────────────────────────────
    print(f"\n--- 1. Alignment of loc embeddings to concept dictionary ---")
    # sample 10K concepts for speed if huge dict
    idx = torch.randperm(K)[:10_000]
    C_sub = C[idx]

    before = (orig @ C_sub.T)
    after  = (centered @ C_sub.T)
    print(f"  BEFORE centering: mean={before.mean():.4f}  std={before.std():.4f}  "
          f"max-per-loc mean={before.max(dim=1).values.mean():.4f}")
    print(f"  AFTER  centering: mean={after.mean():.4f}  std={after.std():.4f}  "
          f"max-per-loc mean={after.max(dim=1).values.mean():.4f}")

    # Positive-cone diagnostic: fraction of loc embeddings that are INSIDE the
    # positive cone of concepts (i.e., CAN be expressed as nonneg combination)
    # Proxy: fraction with max concept cosine > threshold
    for thresh in [0.0, 0.05, 0.1, 0.2]:
        frac = (after.max(dim=1).values > thresh).float().mean().item()
        print(f"  frac loc with max-cos > {thresh:.2f} (after centering): {frac:.4f}")

    # ── 2. Recompose ceiling: best possible R2 given the pipeline ─────────────
    # Even a perfect solver can only produce normalize(normalize(emb-mean) + mean)
    # at best. Check how far this is from the original emb.
    print(f"\n--- 2. Recompose ceiling (perfect weights → how good can we get?) ---")
    recon_ceil = F.normalize(centered + mean, dim=1)   # assumes perfect reconstruction
    metrics(orig, recon_ceil, "ceiling: normalize(normalize(e-m) + m)")
    # Breakdown: how much does ||emb - mean|| vary?
    centered_mag = (orig - mean).norm(dim=1)
    print(f"  ||emb - mean|| : mean={centered_mag.mean():.4f}  "
          f"std={centered_mag.std():.4f}  min={centered_mag.min():.4f}  "
          f"max={centered_mag.max():.4f}")
    # If ||emb-mean|| ≈ 1 for all emb, the ceiling is near-perfect.
    # If it varies a lot, the double-normalize introduces per-sample distortion.

    # ── 3. ADMM baseline (what the saved model does) ──────────────────────────
    print(f"\n--- 3. ADMM baseline ---")
    with torch.no_grad():
        w_admm   = splice.decompose(centered)
        recon_a  = recompose(w_admm, C, mean)
    metrics(orig, recon_a, "ADMM (non-neg + L1)")
    print(f"  avg active concepts : {(w_admm > 0).float().sum(dim=1).mean():.1f} / {K}")
    print(f"  all-zero fraction   : {(w_admm.sum(dim=1) == 0).float().mean():.4f}")

    # ── 4. Oracle: unconstrained min-norm weights ─────────────────────────────
    print(f"\n--- 4. Oracle (unconstrained, no L1, no non-neg) ---")
    with torch.no_grad():
        w_oracle  = oracle_weights(centered, C)
        recon_or  = recompose(w_oracle, C, mean)
    metrics(orig, recon_or, "Oracle (min-norm LS)")
    neg_frac = (w_oracle < 0).float().mean().item()
    print(f"  fraction negative weights : {neg_frac:.4f}")
    if (w_oracle > 0).any():
        pos_mag = w_oracle[w_oracle > 0].abs().mean().item()
        neg_mag = w_oracle[w_oracle < 0].abs().mean().item() if neg_frac > 0 else 0.
        print(f"  |neg|/|pos| mean          : {neg_mag:.4f} / {pos_mag:.4f} = {neg_mag/(pos_mag+1e-9):.3f}")

    # Gap between oracle and ADMM = cost of non-negativity + sparsity constraints
    # Gap between oracle and ceiling = cost of concept coverage

    # ── 4b. Worst-reconstructed locations (oracle cosine) ─────────────────────
    print(f"\n--- 4b. Worst-reconstructed locations (oracle, climplicit only) ---")
    if name == "climplicit":
        cos_per_loc = cos_sim(orig, recon_or).cpu()
        worst_idx   = cos_per_loc.argsort()[:20]
        worst_df    = df.iloc[worst_idx.numpy()].copy()
        worst_df["oracle_cos"] = cos_per_loc[worst_idx].numpy()
        print(worst_df[["lat", "lon", "oracle_cos"]].to_string(index=False))

    # ── 4c. Concepts most needed with negative weight (oracle) ────────────────
    print(f"\n--- 4c. Top concepts by negative-weight demand (oracle) ---")
    if name == "climplicit":
        concepts = torch.load(_p["splice"]["climplicit_geospatial_all"]["concepts"],
                              map_location="cpu", weights_only=False)
        neg_demand = w_oracle.clamp(max=0).abs().mean(dim=0)  # (K,) mean |neg w| per concept
        top_k = neg_demand.argsort(descending=True)[:20]
        for i in top_k:
            print(f"  {concepts[i]:30s}  mean_neg_w={neg_demand[i]:.6f}")

    # ── 5. Oracle with concept-centered dictionary (original SpLiCE style) ────
    print(f"\n--- 5. Oracle with CENTERED concept dictionary ---")
    concept_mean = C.mean(dim=0, keepdim=True)
    C_cent = F.normalize(C - concept_mean, dim=1)
    with torch.no_grad():
        w_oc    = oracle_weights(centered, C_cent)
        recon_oc = recompose(w_oc, C_cent, mean)
    metrics(orig, recon_oc, "Oracle, centered concepts")

    # ── 6. ADMM with concept-centered dictionary ───────────────────────────────
    print(f"\n--- 6. ADMM with CENTERED concept dictionary ---")
    from external.splice.splice.model import SPLICE as SPLICEModel
    splice_cent = SPLICEModel(mean, C_cent, solver='admm',
                              l1_penalty=splice.l1_penalty,
                              return_weights=True, device=DEVICE)
    with torch.no_grad():
        w_ac    = splice_cent.decompose(centered)
        recon_ac = recompose(w_ac, C_cent, mean)
    metrics(orig, recon_ac, "ADMM, centered concepts")
    print(f"  avg active concepts : {(w_ac > 0).float().sum(dim=1).mean():.1f} / {K}")

    # ── 7. Per-dim variance ratio ──────────────────────────────────────────────
    print(f"\n--- 7. Per-dim variance: orig vs ADMM recon ---")
    var_orig  = orig.var(0).mean().item()
    var_recon = recon_a.var(0).mean().item()
    print(f"  var(orig) per dim  : {var_orig:.6f}")
    print(f"  var(recon) per dim : {var_recon:.6f}")
    print(f"  ratio recon/orig   : {var_recon/var_orig:.4f}")
    print(f"  (ratio << 1 means recon collapses to near-constant → R2 << 0)")

    # ── 8. Mean ablation ───────────────────────────────────────────────────────
    print(f"\n--- 8. Mean ablation (image_mean = 0) ---")
    zero_mean = torch.zeros_like(mean)
    splice_nm = SPLICEModel(zero_mean, C, solver='admm',
                            l1_penalty=splice.l1_penalty,
                            return_weights=True, device=DEVICE)
    with torch.no_grad():
        w_nm    = splice_nm.decompose(F.normalize(orig, dim=1))
        recon_nm = recompose(w_nm, C, zero_mean)
    metrics(orig, recon_nm, "ADMM, mean=0")

print("\nDone.")

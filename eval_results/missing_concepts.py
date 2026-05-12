"""
Identify concept dictionary gaps for Climplicit SpLiCE reconstruction.

Modes (--mode):
  residual  : oracle residuals from actual location embeddings → PCA top directions.
              Finds concepts covering empirically important missing variance.
  nullspace : SVD null space of the dictionary matrix C.
              Finds concepts covering structurally uncovered embedding dimensions.
  both      : run both and union-rank results.
"""
import sys
sys.path.insert(0, "..")

import argparse
import json
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path

from splice import _LocWrapper  # noqa: F401 (needed for torch.load)
from paths import load_paths
from models.model import build_model
from models.finetune import apply_lora

BATCH = 256


def load_climplicit_model(model_path: str, device: str):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    a = argparse.Namespace(**ckpt["args"])
    model = build_model(
        text_encoder=a.text_encoder,
        location_encoder=a.location_encoder,
        text_projection=a.text_projection,
        location_projection=a.location_projection,
        shared_dim=a.shared_dim,
        text_finetune_mode=a.text_finetune_mode,
        loc_finetune_mode=a.loc_finetune_mode,
        text_proj_hidden_layers=a.text_proj_hidden_layers,
        text_proj_hidden_features=a.text_proj_hidden_features,
        loc_proj_hidden_layers=a.loc_proj_hidden_layers,
        loc_proj_hidden_features=a.loc_proj_hidden_features,
        text_nonlinearity=a.text_nonlinearity,
        loc_nonlinearity=a.loc_nonlinearity,
        precomputed_text_embeddings=False,
        precomputed_location_embeddings=False,
        device=device,
    )
    if a.text_finetune_mode == "lora":
        model.text_encoder.text_encoder.m = apply_lora(
            model.text_encoder.text_encoder.m, a.lora_rank
        )
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval()


def oracle_weights(centered, C):
    return centered @ torch.linalg.pinv(C.T @ C) @ C.T


def main():
    p_args = argparse.ArgumentParser()
    p_args.add_argument("--mode",        choices=["residual", "nullspace", "both"], default="residual")
    p_args.add_argument("--n_samples",   type=int, default=2000)
    p_args.add_argument("--n_dirs",      type=int, default=10,  help="Top PCA/null-space directions per mode")
    p_args.add_argument("--null_thresh", type=float, default=1e-5, help="Singular-value threshold for null space (relative to s[0])")
    p_args.add_argument("--top_k",       type=int, default=40,  help="Concepts to print")
    p_args.add_argument("--device",      default="cuda")
    p_args.add_argument("--lat_lon_csv", default="/home/libe2152/data/dense_grid/dense_grid.csv")
    args = p_args.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    paths  = load_paths()

    # ── load splice model (for location encoder + dictionary) ─────────────────
    splice = torch.load(
        paths["splice"]["climplicit_geospatial_all"]["model"],
        map_location=device, weights_only=False,
    )
    splice.eval()
    C    = splice.dictionary.to(device)   # (K, D)
    mean = splice.image_mean              # (D,)
    existing = set(torch.load(
        paths["splice"]["climplicit_test_2000"]["concepts"],
        map_location="cpu", weights_only=False,
    ))

    # ── load full climplicit model (for text encoder) ─────────────────────────
    clim_path = paths["models"]["climplicit_open_clip_vit_l"]["model_path"]
    model = load_climplicit_model(clim_path, device)

    # ── direction sets ────────────────────────────────────────────────────────
    dirs_residual = dirs_null = None

    if args.mode in ("residual", "both"):
        df      = pd.read_csv(args.lat_lon_csv).sample(n=args.n_samples, random_state=42)
        latlons = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(device)
        with torch.no_grad():
            orig      = F.normalize(model.location_model_predict(latlons), dim=1)
            centered  = F.normalize(orig - mean, dim=1)
            w_ora     = oracle_weights(centered, C)
            recon_or  = F.normalize(F.normalize(w_ora @ C, dim=1) + mean, dim=1)
            residuals = F.normalize(orig - recon_or, dim=1)   # (N, D)
        cos_per_loc = (orig * recon_or).sum(dim=1)
        print(f"Oracle cosine — mean: {cos_per_loc.mean():.4f}  "
              f"min: {cos_per_loc.min():.4f}  max: {cos_per_loc.max():.4f}")
        _, _, Vh      = torch.linalg.svd(residuals, full_matrices=False)
        dirs_residual = Vh[: args.n_dirs]   # (n_dirs, D)
        print(f"Residual mode: using {dirs_residual.shape[0]} PCA directions from oracle residuals")

    if args.mode in ("nullspace", "both"):
        _, S, Vh  = torch.linalg.svd(C, full_matrices=True)
        rank      = int((S > S[0] * args.null_thresh).sum())
        dirs_null = Vh[rank:]                      # full null space (D - rank, D)
        print(f"Null-space mode: dictionary rank {rank}/{C.shape[1]}, "
              f"{dirs_null.shape[0]} null dimensions")
        if args.n_dirs < dirs_null.shape[0]:
            # keep only the n_dirs most "central" null directions (smallest singular values)
            dirs_null = dirs_null[: args.n_dirs]
            print(f"  (truncated to first {args.n_dirs} null directions; use --n_dirs to change)")

    # ── candidate concepts not in current dictionary ──────────────────────────
    cand_path  = Path("/home/libe2152/projects/explainable-earth-embeddings/data/git-10m/concept_dictionaries/concept_dictionary.json")
    candidates = json.load(open(cand_path))
    new_cands  = [c for c in candidates if c not in existing]
    print(f"Existing dict: {len(existing)}  |  Candidates: {len(candidates)}  |  New: {len(new_cands)}")

    # ── encode candidates with text encoder ───────────────────────────────────
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(new_cands), BATCH):
            e = model.text_model_predict(new_cands[i : i + BATCH])
            all_embs.append(F.normalize(e, dim=1))
    cand_embs = torch.cat(all_embs)   # (M, D)

    # ── score and print ───────────────────────────────────────────────────────
    def score_and_print(dirs, label):
        scores  = (cand_embs @ dirs.T).abs().max(dim=1).values
        top_idx = scores.argsort(descending=True)[: args.top_k]
        print(f"\nTop {args.top_k} missing concepts [{label}]:")
        print(f"  {'concept':<35}  score")
        print(f"  {'-'*35}  -----")
        for i in top_idx:
            print(f"  {new_cands[i]:<35}  {scores[i]:.4f}")
        return scores

    if args.mode == "residual":
        score_and_print(dirs_residual, "residual")
    elif args.mode == "nullspace":
        score_and_print(dirs_null, "nullspace")
    else:  # both — union rank by max of the two scores
        s_res  = score_and_print(dirs_residual, "residual")
        s_null = score_and_print(dirs_null,     "nullspace")
        combined = torch.stack([s_res, s_null]).max(dim=0).values
        top_idx  = combined.argsort(descending=True)[: args.top_k]
        print(f"\nTop {args.top_k} missing concepts [union / max score]:")
        print(f"  {'concept':<35}  combined  residual  nullspace")
        print(f"  {'-'*35}  --------  --------  ---------")
        for i in top_idx:
            print(f"  \"{new_cands[i]}\",",)#  {combined[i]:.4f}    {s_res[i]:.4f}    {s_null[i]:.4f}")


if __name__ == "__main__":
    main()

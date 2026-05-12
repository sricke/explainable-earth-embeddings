import argparse
import json
from pathlib import Path

import sys
sys.path.append("..")
from splice import _LocWrapper
from paths import DATA_ROOT
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score

BATCH_SIZE = 512


def _sparse_emb_path(splice_model_path: str, lat_lon_csv: str, num_samples=None, seed=0) -> Path:
    p = Path(splice_model_path)
    model_stem   = p.parent.parent.name
    concept_stem = p.parent.name
    csv_stem     = Path(lat_lon_csv).stem
    suffix = f"__n{num_samples}_s{seed}" if num_samples is not None else ""
    return DATA_ROOT / "splice_embeddings" / f"{model_stem}__{concept_stem}__{csv_stem}{suffix}.pt"


def evaluate(splice_model_path: str, lat_lon_csv: str, device: str, output: str, num_samples=None, seed=0):
    device = device if torch.cuda.is_available() else "cpu"

    cache_path = _sparse_emb_path(splice_model_path, lat_lon_csv, num_samples, seed)

    splice = torch.load(splice_model_path, map_location=device, weights_only=False)
    print(f"SpLiCE solver = {splice.solver}")
    splice.eval()
    pca = getattr(splice, 'pca', None)

    # if cache_path.exists():
    #     print(f"Loading cached sparse embeddings from {cache_path}")
    #     cached   = torch.load(cache_path, map_location="cpu", weights_only=True)
    #     orig_t   = cached["orig"]
    #     sparse_t = cached["sparse"]
    #     with torch.no_grad():
    #         recon_t = splice.recompose_image(sparse_t.to(device)).cpu()
    # else:
    df = pd.read_csv(lat_lon_csv)
    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=seed)

    latlon = torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(device)

    all_orig, all_sparse, all_recon = [], [], []
    print(f"SpLiCE location mean embedding norm: {splice.image_mean.norm()}")
    # print(f"SpLiCE text mean embedding norm: {splice.text_mean.norm()}")
    with torch.no_grad():
        for i in tqdm(range(0, len(latlon), BATCH_SIZE), desc="Iterating through latlons..."):
            raw = splice.clip.encode_image(latlon[i:i + BATCH_SIZE])
            if pca is not None:
                raw = torch.tensor(pca.transform(raw.cpu().numpy()), dtype=torch.float32).to(device)
            emb     = F.normalize(raw, dim=1)
            weights = splice.decompose(F.normalize(emb - splice.image_mean, dim=1))
            all_orig.append(emb.cpu())
            all_sparse.append(weights.cpu())
            all_recon.append(splice.recompose_image(weights).cpu())

    orig_t   = torch.cat(all_orig)
    sparse_t = torch.cat(all_sparse)
    recon_t  = torch.cat(all_recon)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"orig": orig_t, "sparse": sparse_t}, cache_path)
    print(f"Saved sparse embeddings to {cache_path}")

    orig  = orig_t.numpy()
    recon = recon_t.numpy()
    cos_sim = (recon * orig).sum(axis=1)

    report = {
        "splice_model": str(splice_model_path),
        "lat_lon_csv": str(lat_lon_csv),
        "n_samples": int(len(orig)),

        "mse": float(mean_squared_error(orig, recon)),
        "r2": float(r2_score(orig, recon, multioutput="uniform_average")),

        "cosine_sim_min": float(cos_sim.min()),
        "cosine_sim_max": float(cos_sim.max()),
        "cosine_sim_mean": float(cos_sim.mean()),

        "avg_active_concepts": float(
            (sparse_t > 0).float().sum(axis=1).mean()
        ),

        "frac_all_zero_solutions": float(
            (sparse_t.sum(axis=1) == 0).float().mean()
        )
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"MSE: {report['mse']:.6f}")
    print(f"R2:  {report['r2']:.6f}")
    print(f"Saved to {out_path}")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splice_model", required=True, help="Path to saved splice_model.pt")
    p.add_argument("--lat_lon_csv",  required=True, help="CSV with 'lat','lon' columns")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--output",       required=True, help="Path to save JSON report")
    p.add_argument("--num_samples",  type=int, default=None)
    p.add_argument("--seed",         type=int, default=0)
    args = p.parse_args()
    evaluate(args.splice_model, args.lat_lon_csv, args.device, args.output, args.num_samples, args.seed)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot distance distribution: for each concept, avg cosine distance to its N closest location embeddings."""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("..")
from models.model import build_model
from models.finetune import apply_lora


def load_model(ckpt_path, device, loc_precomputed=True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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
        precomputed_location_embeddings=loc_precomputed,
        device=device,
    )
    if a.text_finetune_mode == 'lora':
        model.text_encoder.text_encoder.m = apply_lora(model.text_encoder.text_encoder.m, a.lora_rank)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), a


@torch.no_grad()
def embed_locations(model, df, device, batch_size=4096):
    embs = torch.from_numpy(np.array(df["location_embedding"].tolist(), dtype=np.float32).copy()) \
        if model.location_encoder.precomputed \
        else torch.from_numpy(df[["lat", "lon"]].to_numpy(dtype=np.float32, copy=True))
    return torch.cat([model.location_model(b.to(device)) for b in tqdm(DataLoader(embs, batch_size), desc="Loc emb")]).cpu()


@torch.no_grad()
def embed_concepts(model, concepts, batch_size=4096):
    return torch.cat([model.text_model_predict(concepts[i:i+batch_size])
                      for i in tqdm(range(0, len(concepts), batch_size), desc="Concept emb")]).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--concept_set", required=True)
    parser.add_argument("--locations_path", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=10, help="Avg distance to N closest locations")
    parser.add_argument("--output", default="plots/distances.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc_p = Path(args.locations_path) if args.locations_path else None
    if loc_p and loc_p.is_dir():
        files = sorted(loc_p.glob("*.parquet")) or sorted(loc_p.glob("*.csv"))
    elif loc_p:
        files = [loc_p]
    else:
        ckpt_dataset = argparse.Namespace(**torch.load(args.model_path, map_location="cpu", weights_only=False)["args"]).dataset_path
        files = sorted(Path(ckpt_dataset).glob("*.parquet"))
    dfs = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if args.num_samples: df = df.sample(args.num_samples, random_state=0).reset_index(drop=True)

    loc_precomputed = "location_embedding" in df.columns
    model, _ = load_model(args.model_path, device, loc_precomputed)

    loc_emb = F.normalize(embed_locations(model, df, device), dim=1)  # (L, D)

    concepts = json.loads(Path(args.concept_set).read_text())
    concepts = list(concepts.keys()) if isinstance(concepts, dict) else [str(c) for c in concepts]
    con_emb = F.normalize(embed_concepts(model, [f"a satellite image of {c}" for c in concepts]), dim=1)  # (C, D)

    # cosine distance = 1 - cosine similarity
    sims = con_emb @ loc_emb.T  # (C, L)
    top_sims = sims.topk(args.top_k, dim=1).values  # (C, top_k)
    avg_dist = (1 - top_sims.mean(dim=1)).numpy()  # (C,)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.hist(avg_dist, bins=50, density=True, edgecolor="none", alpha=0.8)
    plt.xlabel(f"Avg cosine distance to top-{args.top_k} locations")
    plt.ylabel("Density")
    plt.title(f"Distance distribution — {Path(args.concept_set).stem}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

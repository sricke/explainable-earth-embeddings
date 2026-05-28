import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

SPLICE_DIR = Path(__file__).parent
sys.path.insert(0, str(SPLICE_DIR / "location_text_alignment"))

from model import SPLICE
from models.model import build_model

DEFAULT_CONCEPT_SET = (
    SPLICE_DIR
    / "location_text_alignment"
    / "concept_dictionaries"
    / "git10m_concept_dictionary.json"
)
PROMPT = "An image of a {concept}"
BATCH_SIZE = 512


class _LocWrapper(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def encode_image(self, x):
        return self.enc(x)


def load_model(model_path, device):
    ckpt = torch.load(Path(model_path), map_location=device)
    a = ckpt["args"]
    model = build_model(
        text_encoder=a.get("text_encoder"),
        location_encoder=a.get("location_encoder"),
        text_projection=a.get("text_projection"),
        shared_dim=a.get("shared_dim"),
        text_finetune_mode=a.get("text_finetune_mode"),
        loc_finetune_mode=a.get("loc_finetune_mode"),
        text_proj_hidden_layers=a.get("text_proj_hidden_layers"),
        text_proj_hidden_features=a.get("text_proj_hidden_features"),
        text_nonlinearity=a.get("text_nonlinearity"),
        precomputed_text_embeddings=False,
        precomputed_location_embeddings=False,
        device=device,
        lora_r=a.get("lora_rank"),
        lora_last_n_layers=a.get("lora_layers"),
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def load_concept_embeddings(model, concept_set, device):
    with open(concept_set) as f:
        concepts = json.load(f)
    prompted = [PROMPT.format(concept=c) for c in concepts]
    with torch.no_grad():
        emb_concepts = torch.cat(
            [
                model.text_model_predict(prompted[i : i + BATCH_SIZE])
                for i in range(0, len(prompted), BATCH_SIZE)
            ]
        )
    return emb_concepts


def load_latlons(lat_lon_csv, device):
    df = pd.read_csv(lat_lon_csv).reset_index(drop=True)
    return torch.tensor(df[["lat", "lon"]].values, dtype=torch.float32).to(device)


def build_splice(model, emb_concepts, latlons, l1_penalty, device):
    with torch.no_grad():
        loc_embs = model.location_model_predict(latlons)
    return SPLICE(
        image_mean=loc_embs.mean(0),
        dictionary=F.normalize(emb_concepts - emb_concepts.mean(0), dim=1),
        clip=_LocWrapper(model.location_encoder).to(device),
        solver="admm",
        device=device,
        return_weights=True,
        return_cosine=True,
        l1_penalty=l1_penalty,
        text_mean=None,
    )


def compute_reconstruction(splice_model, latlons):
    all_orig, all_weights, all_recon = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(latlons), BATCH_SIZE)):
            raw = splice_model.clip.encode_image(latlons[i : i + BATCH_SIZE])
            orig_b = F.normalize(raw, dim=1)
            weights_b = splice_model.decompose(
                F.normalize(orig_b - splice_model.image_mean, dim=1)
            )
            all_orig.append(orig_b)
            all_weights.append(weights_b)
            all_recon.append(splice_model.recompose_image(weights_b))
    return torch.cat(all_orig), torch.cat(all_weights), torch.cat(all_recon)


def compute_metrics(orig, weights, recon, args):
    return {
        "model_path": args.model_path,
        "lat_lon_csv": args.lat_lon_csv,
        "n_samples": len(orig),
        "l1_penalty": args.l1_penalty,
        "mse": F.mse_loss(recon, orig).item(),
        "cosine_sim": (recon * orig).sum(dim=1).mean().item(),
        "mean_active_concepts": (weights > 0).float().sum(dim=1).mean().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to checkpoint")
    p.add_argument("--lat_lon_csv", required=True)
    p.add_argument("--output", required=False, help="Path to save JSON report")
    p.add_argument(
        "--concept_set",
        default=str(DEFAULT_CONCEPT_SET),
        help="Path to concept dictionary JSON",
    )
    p.add_argument("--l1_penalty", type=float, default=0.175)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = load_model(args.model_path, args.device)
    emb_concepts = load_concept_embeddings(model, args.concept_set, args.device)
    latlons = load_latlons(args.lat_lon_csv, args.device)
    splice_model = build_splice(
        model, emb_concepts, latlons, args.l1_penalty, args.device
    )
    orig, weights, recon = compute_reconstruction(splice_model, latlons)
    report = compute_metrics(orig, weights, recon, args)

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

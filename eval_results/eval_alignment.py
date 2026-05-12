import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GeoTextDataset
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
    if a.text_finetune_mode == "lora":
        model.text_encoder.text_encoder.m = apply_lora(model.text_encoder.text_encoder.m, a.lora_rank)
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval(), a


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    print("Computing cosine similarity...")
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def _uniformity(emb: torch.Tensor, t: float = 2.0, max_samples: int = 8192) -> float:
    """Wang & Isola 2020 uniformity loss. Lower = more uniform."""
    print("Computing uniformity...")
    if emb.shape[0] > max_samples:
        emb = emb[torch.randperm(emb.shape[0], device=emb.device)[:max_samples]]
    sq_dist = 2 - 2 * (emb @ emb.T)
    mask = ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    return sq_dist[mask].mul(-t).exp().mean().log().item()


@torch.no_grad()
def evaluate(model_path: str, dataset_path: str | None, split: str, device: str, batch_size: int, num_samples: int | None, output: str):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location=device)
    saved_args = argparse.Namespace(**ckpt["args"])

    if dataset_path is None:
        dataset_path = saved_args.dataset_path

    model, _ = load_model(model_path, device, loc_precomputed=getattr(saved_args, "precomputed_location_embeddings", True))

    dataset = GeoTextDataset(
        root=dataset_path,
        split=split,
        precomputed_text_embeddings=getattr(saved_args, "precomputed_text_embeddings", True),
        precomputed_location_embeddings=getattr(saved_args, "precomputed_location_embeddings", True),
    )
    if num_samples is not None and num_samples < len(dataset):
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_text, all_loc = [], []
    for locs, texts in tqdm(loader, desc="Iterating through test loader..."):
        if isinstance(texts, torch.Tensor):
            texts = texts.to(device)
        text_emb, loc_emb = model(texts, locs.to(device))
        all_text.append(text_emb.cpu())
        all_loc.append(loc_emb.cpu())

    text_emb = torch.cat(all_text)
    loc_emb  = torch.cat(all_loc)

    print(ckpt)

    report = {
        "checkpoint": model_path,
        "step": ckpt["step"],
        "split": split,
        "n_samples": text_emb.shape[0],
        "cosine_similarity": _cosine_similarity(text_emb, loc_emb),
        "uniformity_text": _uniformity(text_emb),
        "uniformity_loc": _uniformity(loc_emb),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split",        default="test", choices=["train", "val", "test"])
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--batch_size",   type=int, default=512)
    parser.add_argument("--num_samples",  type=int, default=None)
    parser.add_argument("--output",       required=True, help="Path to save JSON report")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.dataset_path, args.split, args.device, args.batch_size, args.num_samples, args.output)

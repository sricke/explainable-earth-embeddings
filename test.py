#!/usr/bin/env python
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from main import Location2TextLightningModule


# -------- config --------
CKPT_PATH = "location2text_pretrained.ckpt"
TOP_K = 10

# Locations as (lat, lon). The model was trained with [lon, lat] order,
# so we convert below when building the tensor.
KNOWN_LOCATIONS = {
    "Paris": (48.8566, 2.3522),
    "New York City": (40.7128, -74.0060),
    "Tokyo": (35.6895, 139.6917),
    "Sydney": (-33.8688, 151.2093),
    "Cairo": (30.0444, 31.2357),
}


def load_model(device: torch.device):
    model = Location2TextLightningModule(
        location_model_type="geoclip",
        location_model=None,
        location_model_filename=None,
        text_model_type="geoclip",
        text_model="geoclip",
        text_vocabulary="openai",
        train_text_model=False,
        learning_rate=1e-4,
        weight_decay=1e-2,
        logit_scale_temperature=0.07,
        lambda_alignment=1.0,
        sigma=1.0,
    )

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def embed_locations(model, device):
    # convert (lat, lon) -> tensor[[lon, lat], ...] to match training
    coords = []
    for name, (lat, lon) in KNOWN_LOCATIONS.items():
        coords.append([lon, lat])
    coords = torch.tensor(coords, dtype=torch.float32, device=device)

    with torch.no_grad():
        loc_emb = model.location_model(coords)
        loc_emb = F.normalize(loc_emb, dim=-1)

    return loc_emb  # shape [N_loc, D]


def embed_concepts(model, device, root: Path):
    vocab_path = root / "text-descriptions-wikipedia" / "mscoco.json"
    with vocab_path.open("r", encoding="utf-8") as f:
        concepts = json.load(f)

    concept_embs = []
    with torch.no_grad():
        for word in concepts:
            emb = model.text_model_predict(word, normalize=True)
            # text_model_predict returns [1, D]
            if emb.ndim == 2:
                emb = emb.squeeze(0)
            concept_embs.append(emb.to(device))

    concept_embs = torch.stack(concept_embs, dim=0)
    concept_embs = F.normalize(concept_embs, dim=-1)
    return concepts, concept_embs  # list[str], [N_concepts, D]


def main():
    root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model and checkpoint...")
    model = load_model(device)

    print("Embedding known locations...")
    loc_emb = embed_locations(model, device)

    print("Embedding geospatial concepts...")
    concepts, concept_embs = embed_concepts(model, device, root)

    # cosine similarities: [N_loc, N_concepts]
    sims = loc_emb @ concept_embs.T

    loc_names = list(KNOWN_LOCATIONS.keys())
    for i, loc_name in enumerate(loc_names):
        lat, lon = KNOWN_LOCATIONS[loc_name]
        print(f"\n=== {loc_name} (lat={lat:.4f}, lon={lon:.4f}) ===")
        top_vals, top_idx = sims[i].topk(TOP_K, dim=-1)
        for score, idx in zip(top_vals.tolist(), top_idx.tolist()):
            print(f"{concepts[idx]:40s}  cos_sim={score:.3f}")


if __name__ == "__main__":
    main()
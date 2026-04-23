import numpy as np
import torch
import torch.nn.functional as F


def get_embeddings(latlons: np.ndarray, location_model, device="cpu", batch_size=256) -> np.ndarray:
    """Return normalized location embeddings (N, dim) without SPLICE decomposition."""
    location_model.eval()
    latlons_t = torch.tensor(latlons, dtype=torch.float32).to(device)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(latlons_t), batch_size):
            emb = location_model(latlons_t[i : i + batch_size])
            all_embs.append(F.normalize(emb, dim=1).cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def get_sparse_codes(latlons: np.ndarray, location_model, splice_model, device="cpu", batch_size=256) -> np.ndarray:
    """Return sparse concept codes (N, num_concepts) for an array of lat/lon pairs."""
    location_model.eval()
    splice_model.eval()
    latlons_t = torch.tensor(latlons, dtype=torch.float32).to(device)
    all_codes = []
    with torch.no_grad():
        for i in range(0, len(latlons_t), batch_size):
            emb = location_model(latlons_t[i : i + batch_size])
            emb = F.normalize(emb, dim=1)
            centered = F.normalize(emb - splice_model.image_mean, dim=1)
            all_codes.append(splice_model.decompose(centered).cpu().numpy())
    return np.concatenate(all_codes, axis=0)

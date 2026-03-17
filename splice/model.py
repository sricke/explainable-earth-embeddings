from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from main import Location2TextLightningModule


def load_model(device: torch.device, ckpt_path: Path | None) -> Location2TextLightningModule:
    """
    Load a trained Location2TextLightningModule from a checkpoint.

    This is backend-agnostic: all hyperparameters (including model types
    and names) are restored from the checkpoint itself.
    """
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No valid checkpoint found for SpLiCE inference: {ckpt_path!r}"
        )

    print(f"[SpLiCE] Loading checkpoint from: {ckpt_path}")
    model = Location2TextLightningModule.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    )
    model.to(device)
    model.eval()
    return model


def compute_mean_location_embedding(
    model: Location2TextLightningModule,
    device: torch.device,
    dataset_csv: Path,
    num_samples: int | None,
) -> torch.Tensor:
    """
    Embed a (subsampled) dataset of locations and return the mean normalized embedding.
    This is used only to estimate the global mean for SpLiCE, not for per-location inference.
    """
    df = pd.read_csv(dataset_csv)
    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=42)
    print(f"[SpLiCE] compute_mean_location_embedding using {len(df)} rows from {dataset_csv}")

    # LocationEmbeddingModel expects [lat, lon] tensors (it will internally
    # swap to [lon, lat] when using a SatCLIP backend). The dataset CSV
    # stores columns as ['lon', 'lat'], so we reorder here to [lat, lon].
    locs_latlon = df[["lat", "lon"]].values.astype("float64")
    print(f"[SpLiCE] First 3 CSV rows (lon, lat): {df[['lon', 'lat']].head(3).to_dict(orient='records')}")
    print(f"[SpLiCE] First 3 tensors fed to location_model (lat, lon): {locs_latlon[:3].tolist()}")
    locs = torch.tensor(locs_latlon, device=device)

    from torch.utils.data import DataLoader

    loader = DataLoader(locs, batch_size=512, shuffle=False)

    running_sum = None
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding locations for mean"):
            embs = model.location_model(batch)
            embs = F.normalize(embs, dim=1)
            if running_sum is None:
                running_sum = embs.sum(dim=0)
            else:
                running_sum += embs.sum(dim=0)
            total += embs.shape[0]
            if total % 5120 == 0:
                torch.cuda.empty_cache()

    mean = running_sum / total
    mean_norm = F.normalize(mean, dim=0)
    print(f"[SpLiCE] Mean embedding norm before normalize: {mean.norm().item():.4f}")
    print(f"[SpLiCE] Mean embedding norm after normalize: {mean_norm.norm().item():.4f}")
    return mean_norm


def embed_known_locations(
    model: Location2TextLightningModule,
    device: torch.device,
    known_locations: dict[str, tuple[float, float]],
):
    """Embed a small set of human-readable named locations."""
    rows = []
    coords = []
    for name, (lat, lon) in known_locations.items():
        rows.append({"fn": name, "lat": lat, "lon": lon})
        coords.append([lat, lon])
    print(f"[SpLiCE] Embedding {len(coords)} known locations")

    df = pd.DataFrame(rows).reset_index(drop=True)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

    with torch.no_grad():
        embeddings = model.location_model(coords_tensor)
        embeddings = F.normalize(embeddings, dim=1)

    return df, embeddings


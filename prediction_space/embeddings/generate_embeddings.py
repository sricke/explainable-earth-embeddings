import torch
import ee
from tqdm import tqdm

import sys
sys.path.append("../../external/satclip")

import satclip


def generate_satclip_embeddings(latlons, device="cuda"):
    from huggingface_hub import hf_hub_download
    from satclip.load import get_satclip

    assert isinstance(latlons, torch.Tensor), "Expected latlons to be a torch.Tensor"
    assert latlons.shape[1] == 2, "Expected latlons shape (N, 2)"

    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
        device=device,
    )  # Only loads location encoder by default
    model.eval()

    emb = torch.zeros((len(latlons), 256), dtype=torch.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(latlons), 1024), desc="Generating SatCLIP embeddings"):
            batch = latlons[i:i+1024].double().to(device)
            emb[i:i+1024] = model(batch).detach().cpu()

    return emb


def generate_geoclip_embeddings(latlons, device="cuda"):
    from geoclip import LocationEncoder

    assert isinstance(latlons, torch.Tensor), "Expected latlons to be a torch.Tensor"
    assert latlons.shape[1] == 2, "Expected latlons shape (N, 2)"

    encoder = LocationEncoder().to(device)
    encoder.eval()

    emb = torch.zeros((len(latlons), 512), dtype=torch.float32)
    with torch.no_grad():
        for i in tqdm(range(0, len(latlons), 1024), desc="Generating GeoCLIP embeddings"):
            batch = latlons[i:i+1024].float().to(device)
            emb[i:i+1024] = encoder(batch).detach().cpu()

    return emb


def generate_aef_embeddings(latlons, batch_size=1000, max_workers=8):
    assert isinstance(latlons, torch.Tensor), "Expected latlons to be a torch.Tensor"
    assert latlons.shape[1] == 2, "Expected latlons shape (N, 2)"

    from concurrent.futures import ThreadPoolExecutor, as_completed

    ae_collection = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filterDate('2022-01-01', '2022-12-31') \
        .mosaic()

    # Determine band order once so all rows use the same column order
    band_names = ae_collection.bandNames().getInfo()

    coords = latlons.cpu().numpy()
    embs = [None] * len(coords)
    nan_row = [float('nan')] * len(band_names)

    def fetch_batch(batch_start):
        batch_coords = coords[batch_start:batch_start + batch_size]
        features = [
            ee.Feature(ee.Geometry.Point(float(lon), float(lat)), {"idx": batch_start + i})
            for i, (lat, lon) in enumerate(batch_coords)
        ]
        fc = ee.FeatureCollection(features)
        samples = ae_collection.sampleRegions(
            collection=fc,
            scale=10,
            geometries=False,
        ).getInfo()
        return batch_start, len(batch_coords), samples

    batch_starts = list(range(0, len(coords), batch_size))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, bs): bs for bs in batch_starts}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating AEF embeddings"):
            batch_start, batch_len, samples = future.result()
            for feature in samples["features"]:
                props = feature["properties"]
                idx = props["idx"]
                row = [props.get(b) for b in band_names]
                embs[idx] = nan_row if any(v is None for v in row) else row
            for i in range(batch_start, batch_start + batch_len):
                if embs[i] is None:
                    embs[i] = nan_row

    return torch.tensor(embs, dtype=torch.float32)
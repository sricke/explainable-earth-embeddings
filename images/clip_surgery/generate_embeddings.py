from torchgeo.datasets import EuroSATSpatial, EuroSAT, BigEarthNet
import rasterio
import torch
from huggingface_hub import hf_hub_download
import numpy as np
import os
import argparse
import torch.nn.functional as F
from typing import ClassVar
import sys
from pathlib import Path
from tqdm import tqdm
from inference_utils import preprocessing_satclip
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_clip_surgery_src = _repo_root / "CLIP_Surgery"
if str(_clip_surgery_src) not in sys.path:
    sys.path.insert(0, str(_clip_surgery_src))
from images.clip_surgery.inference_satclip import get_saliency_mask
import clip
from images.clip_surgery import get_satclip
from images.clip_surgery.inference_utils import (
    RANDOM_LOCATION,
    encode_locations,
    load_image_sentinel,
    surgery_similarity_patch_rows,
)

SUPPORTED_MODELS = {
    "vit16": "microsoft/SatCLIP-ViT16-L40",
    "resnet50": "microsoft/SatCLIP-ResNet50-L40",
}



class BigEarthNetWithLocations(BigEarthNet):
    
    @staticmethod
    def _get_patch_center(path):
        with rasterio.open(path) as src:
            left, bottom, right, top = rasterio.warp.transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )
        # patch center (lon, lat)
        lon = (left + right) / 2
        lat = (bottom + top) / 2
        return lat, lon
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['lat'], sample['lon'] = self._get_location(index)
        # Stable per-patch identifier: the BigEarthNet patch folder name.
        folders = self.folders[index]
        sample['path'] = folders['s2'] if self.bands != 's1' else folders['s1']
        return sample

    def _get_location(self, index):
        # BigEarthNet has no `samples` list; use the first band file's geotransform.
        paths = self._load_paths(index)
        lat, lon = self._get_patch_center(paths[0])
        return lat, lon

class EuroSATSpatialWithLocations(EuroSATSpatial):
      
    @staticmethod
    def _get_patch_center(path):
        with rasterio.open(path) as src:
            left, bottom, right, top = rasterio.warp.transform_bounds(
                src.crs, "EPSG:4326", *src.bounds
            )
        # patch center (lon, lat)
        lon = (left + right) / 2
        lat = (bottom + top) / 2
        return lat, lon
        
    def _get_location(self, index):
        path, _ = self.samples[index]
        lat, lon = self._get_patch_center(path)
        return lat, lon
    
    def __getitem__(self, index):
        image, label = self._load_image(index)
        lat, lon = self._get_location(index)
        image = torch.index_select(image, dim=0, index=self.band_indices).float()
        path = self.samples[index][0]
        sample = {'image': image, 'label': label, 'lat': lat, 'lon': lon, 'path': path}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    
parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    default="eurosat",
    choices=["eurosat", "bigearthnet" ],
    help="Dataset to use.",
)

parser.add_argument(
        "--bbox_percentile",
        type=float,
        default=0.9,
        help="Bbox mask: use max( this quantile of raw sum heatmap, min-peak fraction * max ). "
        "Higher = fewer, stronger regions (default: 0.94).",
)
parser.add_argument(
    "--bbox_min_peak_fraction",
    type=float,
    default=0.5,
    help="Bbox mask: pixel value must be at least this fraction of the tile peak "
    "(combined with --bbox_percentile via max; default: 0.5).",
)
parser.add_argument(
    "--bbox_open_kernel",
    type=int,
    default=3,
    help="Optional opening kernel (odd >=3); 0 disables. Closing is never applied.",
)
args = parser.parse_args()

ok = int(args.bbox_open_kernel)
if ok > 0 and ok % 2 == 0:
    ok += 1
device = "cuda" if torch.cuda.is_available() else "cpu"
satclip_model_path = SUPPORTED_MODELS["vit16"]
satclip_model = get_satclip(
    hf_hub_download(satclip_model_path, "satclip-vit16-l40.ckpt"),
    device=device,
    surgery=True,
    return_all=True,
)
satclip_model.eval()
encode_location = satclip_model.encode_location
redundant_feats = encode_locations([RANDOM_LOCATION], encode_location, device, satclip=True)
out_dir = _repo_root / "out_new" / args.dataset 
out_dir.mkdir(parents=True, exist_ok=True)
locations = []
chunk_size = 50000
for split in ["val", "test", "train"]:
    print(f"Processing {split} split...")
    if args.dataset == "eurosat":
        dataset = EuroSATSpatialWithLocations(split=split, download=True)
    elif args.dataset == "bigearthnet":
        dataset = BigEarthNetWithLocations(split=split, download=True, bands="s2")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    n_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    for i in range(n_chunks):
        results = []
        labels = []
        lats = []
        lons = []
        paths = []
        start = i * chunk_size
        end = min(start + chunk_size, len(dataset))
        for j in tqdm(range(start, end), desc=f"{split} chunk {i}"):
            item = dataset[j]
            image, label, lat, lon, path = (
                item['image'], item['label'], item['lat'], item['lon'], item['path']
            )
            h, w = image.shape[-2:]
            image, _ = preprocessing_satclip(image, device=device)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            with torch.no_grad():
                vm = satclip_model.visual
                final_tokens, layer_hiddens = vm.forward_intermediates(
                    image,
                    indices=None,
                    norm=False,
                    output_fmt="NLC",
                )
                image_features = vm.forward_head(final_tokens, pre_logits=False)
                location_features = encode_locations([(None, lon, lat)], encode_location, device, satclip=True)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (B, 1+ P, D ) P = 196 (14*14)
                similarity = clip.clip_feature_surgery(
                    image_features, location_features, redundant_feats=redundant_feats
                )
                similarity_map = clip.get_similarity_map(
                    surgery_similarity_patch_rows(similarity), (h, w)
                )
            
                sum_hm = None
                for li, hid in enumerate(layer_hiddens[-6:]):
                    feat = vm.forward_head(vm.norm(hid), pre_logits=False)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    sim = clip.clip_feature_surgery(
                        feat, location_features, redundant_feats=redundant_feats
                    )
                    smap = clip.get_similarity_map(
                        surgery_similarity_patch_rows(sim), (h, w)
                    )
                    hm = smap[0, :, :, 0].cpu().numpy()
                    if sum_hm is None:
                        sum_hm = np.zeros_like(hm, dtype=np.float64)
                    sum_hm = sum_hm + hm
                sum_hm_f32: np.ndarray | None = None
                if sum_hm is not None:
                    sum_hm_f32 = sum_hm.astype(np.float32)
                    smin, smax = float(sum_hm.min()), float(sum_hm.max())
                    sum_display = (sum_hm - smin) / (smax - smin + 1e-8)
                    sum_display = sum_display.astype(np.float32)
                    saliency_mask = get_saliency_mask(
                        sum_hm_f32,
                        percentile=float(args.bbox_percentile),
                        min_peak_fraction=float(args.bbox_min_peak_fraction),
                        open_kernel_size=ok,
                    )

                    # Build token-space attention bias: SDPA wants shape broadcastable to
                    # (B, H, L, S) where L=S=1+(H/p)*(W/p). Downsample the pixel saliency
                    # mask to the patch grid, add a CLS slot, and turn it into an additive
                    # bias (0 = attend, -inf = mask) of shape (B, 1, 1, S).
                    patch_size = vm.patch_embed.patch_size
                    ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
                    gh, gw = image.shape[-2] // ph, image.shape[-1] // pw  # (H_n/p, W_n/p) -> 224/16 -> (14, 14)
                    mask_t = torch.from_numpy(saliency_mask).to(image.device).float()# (H, W) -> (64, 64)
                    mask_patches = F.interpolate(
                        mask_t[None, None], size=(gh, gw), mode="area"
                    ) # mask_t is (64, 64) and then interpolated to (14, 14) -> average of ~4 pixels
                    # mask patches that are not attended
                    mask_patches = (mask_patches > 0).float().flatten(1)  # (1, gh*gw) -> (1, 196)
                    cls = torch.ones(mask_patches.shape[0], 1, device=mask_patches.device)
                    key_mask = torch.cat([cls, mask_patches], dim=1)  # (1, 1+gh*gw) -> (1, 197)
                    attn_bias = (key_mask - 1.0) * 1e9
                    attn_bias = attn_bias[:, None, None, :]  # (B, 1, 1, S) -> (1, 1, 1, 197)
                    final_tokens, layer_hiddens = vm.forward_intermediates(
                        image,
                        attn_mask=attn_bias,
                        indices=None,
                        norm=False,
                        output_fmt="NLC",
                    )
                    cls_token = final_tokens[:, 0, :]
                    masked_features = vm.forward_head(cls_token, pre_logits=False)
                    masked_features = masked_features / masked_features.norm(
                        dim=-1, keepdim=True
                    )
                    labels.append(np.asarray(label))
                    lats.append(float(lat))
                    lons.append(float(lon))
                    paths.append(str(path))
                    results.append(masked_features.cpu().numpy())
        results = np.concatenate(results, axis=0)
        labels = np.stack(labels)
        lats = np.asarray(lats, dtype=np.float32)
        lons = np.asarray(lons, dtype=np.float32)
        paths = np.asarray(paths)
        print(
            f"Saving {split} chunk {i}: {results.shape[0]} embeddings, "
            f"labels {labels.shape}, paths {paths.shape}"
        )
        np.savez_compressed(
            os.path.join(out_dir, f"{split}_masked_{i:04d}.npz"),
            embeddings=results,
            labels=labels,
            lat=lats,
            lon=lons,
            paths=paths,
        )


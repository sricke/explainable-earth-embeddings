"""GeoCLIP CLIP-Surgery inference (RGB images)."""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_clip_surgery_src = _repo_root / "CLIP_Surgery"
if str(_clip_surgery_src) not in sys.path:
    sys.path.insert(0, str(_clip_surgery_src))

import cv2
import numpy as np
import torch
from tqdm import tqdm

import clip
from images.clip_surgery.geoclip_surgery.load import get_geoclip
from images.clip_surgery.inference_utils import (
    encode_locations,
    load_image_geoclip,
    load_rgb_image,
    load_s2_sample,
    plot_layer_similarity_grid,
    safe_results_subdir,
    save_surgery_maps_to_png,
    surgery_similarity_patch_rows,
    RANDOM_LOCATION,
)

IM2GPS_CSV = "/home/seri6958/translocator_eval_data/im2gps_places365.csv"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="GeoCLIP CLIP surgery maps for RGB images.")
    parser.add_argument("--architecture", choices=["vit16", "resnet50"], default="vit16")
    parser.add_argument("--source_folder", type=str, default="/home/seri6958/Places_from_im2gps")
    parser.add_argument(
        "--no_surgery",
        action="store_true",
        help="Disable surgery (per-layer similarity disabled).",
    )
    parser.add_argument(
        "--no_layer_grid",
        action="store_true",
        help="Skip per-layer similarity grid ({id}_layers.png).",
    )
    parser.add_argument(
        "--layer_grid_normalize",
        choices=["none", "minmax"],
        default="minmax",
        help="Per-panel heatmap scaling for layers.png / sum_layers.png: "
        "'none' is legacy (hm*255); 'minmax' stretches each panel to full colormap range.",
    )
    args = parser.parse_args()

    use_surgery = not args.no_surgery
    layer_grid = not args.no_layer_grid and use_surgery
    layer_grid_norm = args.layer_grid_normalize

    if args.architecture == "resnet50":
        raise ValueError("Resnet50 not implemented yet")

    geo_model = get_geoclip(device=device, surgery=use_surgery, return_all=True)
    geo_model.eval()
    encode_location = geo_model.location_encoder.forward

    all_locations = load_s2_sample(IM2GPS_CSV, "IMG_ID", lon_column="LON", lat_column="LAT")
    all_location_features = encode_locations(
        all_locations, encode_location, device, satclip=False
    )
    id_to_idx = {loc[0]: idx for idx, loc in enumerate(all_locations)}
    redundant_feats = encode_locations([RANDOM_LOCATION], encode_location, device, satclip=False)
    places_folder = args.source_folder
    data_folder = "images"

    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]

    for place in places:
        index_path = os.path.join(place, "index.csv")
        all_locations_place = load_s2_sample(index_path, "id")
        place_name = Path(place).name
        results_dir = _repo_root / "out" / "geoclip" / place_name
        results_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        print(f"Computing geoclip surgery maps for {place}...")
        similarity_maps_dict = {}
        rgb_images_dict = {}
        cv2_img_bgr_dict = {}

        vm = geo_model.image_encoder.CLIP.vision_model
        vproj = geo_model.image_encoder.CLIP.visual_projection
        mlp = geo_model.image_encoder.mlp

        for id_val, lon, lat in tqdm(all_locations_place, desc=f"CLIP Surgery {place}"):
            n = id_to_idx.get(id_val)
            if n is None:
                continue
            data_path = os.path.join(place, data_folder, id_val)
            try:
                image = load_image_geoclip(data_path, device=device)
                rgb_img = load_rgb_image(data_path)
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue

            out_dir = os.path.join(results_dir, safe_results_subdir(id_val))
            os.makedirs(out_dir, exist_ok=True)

            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                if layer_grid:
                    vision_out, layer_hiddens = vm.forward_intermediates(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                    )
                else:
                    layer_hiddens = None
                    vision_out = vm(image)
                    breakpoint()
                patch_tokens = vproj(vm.post_layernorm(vision_out.last_hidden_state))
                image_features = mlp(patch_tokens)
                breakpoint()

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = clip.clip_feature_surgery(
                    image_features, all_location_features, redundant_feats=redundant_feats
                )
                similarity_map = clip.get_similarity_map(
                    surgery_similarity_patch_rows(similarity), (h, w)
                )

            similarity_maps_dict[id_val] = similarity_map[:, :, :, n].cpu()
            rgb_images_dict[id_val] = rgb_img
            cv2_img_bgr_dict[id_val] = cv2_img_bgr

            if layer_grid and layer_hiddens is not None:
                with torch.no_grad():
                    heatmaps = []
                    titles = []
                    sum_hm = None
                    for li, hid in enumerate(layer_hiddens[-6:]):
                        feat = mlp(vproj(vm.post_layernorm(hid)))
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        sim = clip.clip_feature_surgery(
                            feat, all_location_features, redundant_feats=redundant_feats
                        )
                        smap = clip.get_similarity_map(
                            surgery_similarity_patch_rows(sim), (h, w)
                        )
                        hm = smap[0, :, :, n].cpu().numpy()
                        heatmaps.append(hm)
                        titles.append(
                            f"L{len(layer_hiddens) - min(6, len(layer_hiddens)) + li}"
                        )
                        if sum_hm is None:
                            sum_hm = np.zeros_like(hm, dtype=np.float64)
                        sum_hm = sum_hm + hm
                    sum_display = None
                    if sum_hm is not None:
                        smin, smax = float(sum_hm.min()), float(sum_hm.max())
                        sum_display = (sum_hm - smin) / (smax - smin + 1e-8)
                        sum_display = sum_display.astype(np.float32)
                plot_layer_similarity_grid(
                    heatmaps,
                    cv2_img_bgr,
                    os.path.join(out_dir, "layers.png"),
                    titles=titles,
                    suptitle=f"geoclip per-layer — idx {n}",
                    ncols=4,
                    heatmap_normalize=layer_grid_norm,
                )
                if sum_display is not None:
                    plot_layer_similarity_grid(
                        [sum_display],
                        cv2_img_bgr,
                        os.path.join(out_dir, "sum_layers.png"),
                        titles=[""],
                        suptitle="",
                        ncols=1,
                        figsize=(8.0, 8.0),
                        heatmap_normalize=layer_grid_norm,
                    )

        save_surgery_maps_to_png(place, results_dir, similarity_maps_dict, cv2_img_bgr_dict)


if __name__ == "__main__":
    main()

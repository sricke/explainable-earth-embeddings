import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from CLIP_Surgery import clip
from geoclip_surgery import get_geoclip
from inference_utils import (
    Location,
    encode_locations,
    load_image_geoclip,
    load_locations,
    plot_similarity_maps,
    seed_everything,
)
from PIL import Image
from tqdm import tqdm

IMAGES_FOLDER = "/data/im2gps300/Places_from_im2gps"
SEED = 42
_THIS_DIR = Path(__file__).resolve().parent


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="GeoCLIP CLIP surgery maps for RGB images.")
    parser.add_argument("--lon_column", type=str, default="lon")
    parser.add_argument("--lat_column", type=str, default="lat")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--source_folder", type=str, default=IMAGES_FOLDER)
    parser.add_argument(
        "--no_surgery",
        action="store_true",
        help="Disable surgery (per-layer similarity disabled).",
    )
    parser.add_argument(
        "--no_layer_grid",
        action="store_true",
        help="Skip per-layer similarity grid; defaults to original surgery",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "minmax"],
        default="minmax",
        help="Per-panel heatmap scaling for layers.png / sum_layers.png: "
        "'none' is legacy (hm*255); 'minmax' stretches each panel to full colormap range.",
    )
    args = parser.parse_args()

    seed_everything(SEED)
    use_surgery = not args.no_surgery
    layer_grid = not args.no_layer_grid and use_surgery
    norm = args.normalize

    geo_model = get_geoclip(device=device, surgery=use_surgery, return_all=True)
    geo_model.eval()
    encode_location = geo_model.location_encoder.forward

    redundant_feats = None
    if use_surgery:
        redundant_feats = encode_locations(
            [Location(id="north_pole", lon=0, lat=90)],
            encode_location,
            device,
            satclip=False,
        )
    places_folder = args.source_folder
    data_folder = "images"

    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]

    for place in places:
        index_path = os.path.join(place, "index.csv")
        all_locations_place = load_locations(
            index_path,
            lon_column=args.lon_column,
            lat_column=args.lat_column,
            id_column=args.id_column,
        )
        place_name = Path(place).name
        results_dir = (
            _THIS_DIR
            / "out"
            / ("geoclip_no_surgery" if not use_surgery else "geoclip_surgery")
            / place_name
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Computing geoclip surgery maps for {place}...")

        vm = geo_model.image_encoder.CLIP.vision_model
        vproj = geo_model.image_encoder.CLIP.visual_projection
        mlp = geo_model.image_encoder.mlp

        for location in tqdm(all_locations_place, desc=f"CLIP Surgery {place}"):
            data_path = os.path.join(place, data_folder, str(location.id))
            target_feat = encode_locations(
                [location], encode_location, device, satclip=False
            )

            image = load_image_geoclip(data_path, device=device)
            rgb_img = np.array(Image.open(data_path).convert("RGB"))

            out_dir = os.path.join(results_dir, location.id)
            os.makedirs(out_dir, exist_ok=True)

            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                if use_surgery:
                    vision_out, layer_hiddens = vm.forward_intermediates(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                    )
                else:
                    vision_out = vm(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                    )
                    layer_hiddens = []

                patch_tokens = vproj(vm.post_layernorm(vision_out.last_hidden_state))
                image_features = mlp(patch_tokens)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                if use_surgery:
                    similarity = clip.clip_feature_surgery(
                        image_features, target_feat, redundant_feats=redundant_feats
                    )
                else:
                    similarity = image_features @ target_feat.t()

                visual_patches_sim = similarity[:, 1:, :] # strip CLS token 
                similarity_map = clip.get_similarity_map(visual_patches_sim, (h, w))

            if not layer_grid:
                plot_similarity_maps(
                    similarity_map[0, :, :, 0].cpu().numpy(),
                    cv2_img_bgr,
                    os.path.join(out_dir, "layers.png"),
                    titles=[],
                    suptitle="",
                    ncols=1,
                    heatmap_normalize=norm,
                )
            else:
                with torch.no_grad():
                    heatmaps = []
                    titles = []
                    sum_hm = None
                    for li, hid in enumerate(layer_hiddens[-6:]):
                        feat = mlp(vproj(vm.post_layernorm(hid)))
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        similarity = clip.clip_feature_surgery(
                            feat, target_feat, redundant_feats=redundant_feats
                        )
                        visual_patches_sim = similarity[:, 1:, :] # strip CLS token 
                        similarity_map = clip.get_similarity_map(visual_patches_sim, (h, w))
                        hm = similarity_map[0, :, :, 0].cpu().numpy()
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

                plot_similarity_maps(
                    heatmaps,
                    cv2_img_bgr,
                    os.path.join(out_dir, "layers.png"),
                    titles=titles,
                    suptitle=f"geoclip per-layer — {location.id}",
                    ncols=4,
                    heatmap_normalize=norm,
                )

                plot_similarity_maps(
                    [sum_display],
                    cv2_img_bgr,
                    os.path.join(out_dir, "sum_layers.png"),
                    titles=[""],
                    suptitle="",
                    ncols=1,
                    figsize=(8.0, 8.0),
                    heatmap_normalize=norm,
                )


if __name__ == "__main__":
    main()

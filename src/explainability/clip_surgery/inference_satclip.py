import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image

from _path_setup import _THIS_DIR

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from CLIP_Surgery import clip
from satclip_surgery.load import get_satclip
from inference_utils import (
    Location,
    seed_everything,
    encode_locations,
    load_image_sentinel,
    load_locations,
    plot_similarity_maps,
    surgery_similarity_patch_rows,
    get_saliency_mask,
    load_rgb_from_images_corr,
    draw_boxes_on_rgb,
    boxes_from_saliency,
    _minmax_normalize,
)

SEED = 42

SUPPORTED_MODELS = {
    "vit16": "microsoft/SatCLIP-ViT16-L40",
}

PLACES_FOLDER = "/data/cities50"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="SatCLIP CLIP surgery maps for Sentinel-2 tiles.")
    
    parser.add_argument("--source_folder", type=str, default=PLACES_FOLDER)
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
        "'none' matches legacy (hm*255); 'minmax' stretches each panel to full colormap range.",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Detect salient regions on summed heatmap; save bbox.png and mask.png.",
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
        "--bbox_max_area",
        type=int,
        default=200,
        help="Maximum connected-component area in pixels for bbox (default: 400).",
    )
    parser.add_argument(
        "--bbox_open_kernel",
        type=int,
        default=3,
        help="Optional opening kernel (odd >=3); 0 disables. Closing is never applied.",
    )
    parser.add_argument(
        "--bbox_square_side",
        type=int,
        default=32,
        help="Fixed square side in pixels for every bbox (default: None).",
    )
    parser.add_argument(
        "--bbox_kernel_size",
        type=int,
        default=None,
        help="Deprecated: use --bbox_open_kernel. If set, overrides --bbox_open_kernel.",
    )
    parser.add_argument(
        "--bbox_line_thickness",
        type=int,
        default=2,
        help="Bounding-box line thickness (default: 2).",
    )
    
    args = parser.parse_args()
    seed_everything(SEED)

    if args.bbox_kernel_size is not None:
        args.bbox_open_kernel = int(args.bbox_kernel_size)

    ok = int(args.bbox_open_kernel)
    if ok > 0 and ok % 2 == 0:
        ok += 1

    use_surgery = not args.no_surgery
    layer_grid = not args.no_layer_grid and use_surgery
    norm = args.normalize

    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
        device=device,
        surgery=use_surgery,
        return_all=True,
    )
    model.eval()
    encode_location = model.encode_location

    places_folder = args.source_folder
    data_folder = "data"

    places = sorted(
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    )
    
    # to emulate noise
    NORTH_POLE_LOCATION = Location(id="north_pole", lon=0, lat=90)

    for place in places:
        
        index_path = os.path.join(place, "index.csv")
        all_locations = load_locations(index_path, lon_column="lon", lat_column="lat", id_column="id")
        all_location_features = encode_locations(
            all_locations, encode_location, device, satclip=True
        )
        id_to_idx = {loc.id: idx for idx, loc in enumerate(all_locations)}
        redundant_feats = encode_locations([NORTH_POLE_LOCATION], encode_location, device, satclip=True)

        place_name = Path(place).name
        results_dir = _THIS_DIR / "out" / "satclip" / place_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Computing satclip surgery maps for {place}...")

        for location in tqdm(all_locations, desc=f"CLIP Surgery {place}"):
            n = id_to_idx.get(location.id)
            if n is None:
                continue
            data_path = os.path.join(place, data_folder, location.id)
            image, _ = load_image_sentinel(data_path, device=device)
            rgb_img = load_rgb_from_images_corr(place, location.id)

            out_dir = os.path.join(str(results_dir), location.id)
            os.makedirs(out_dir, exist_ok=True)

            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                vm = model.visual
            
                final_tokens, layer_hiddens = vm.forward_intermediates(
                    image,
                    indices=None,
                    norm=False,
                    output_fmt="NLC",
                )
                image_features = vm.forward_head(final_tokens, pre_logits=False)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                similarity = clip.clip_feature_surgery(
                    image_features, all_location_features, redundant_feats=redundant_feats
                )
                similarity_map = clip.get_similarity_map(
                    surgery_similarity_patch_rows(similarity), (h, w)
                )
                
            if not layer_grid: # original surgery map
                plot_similarity_maps(
                    similarity_map[0, :, :, n].cpu().numpy(),
                    cv2_img_bgr,
                    os.path.join(out_dir, f"layers_{location.id}.png"),
                    titles=["Original Surgery"],
                    suptitle="",
                    ncols=1,
                )

            else:
                # compute per layer similarity maps
                with torch.no_grad():
                    heatmaps = []
                    titles = []
                    sum_hm = None
                    for li, hid in enumerate(layer_hiddens[-6:]):
                        feat = vm.forward_head(vm.norm(hid), pre_logits=False)
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
                    sum_hm_f32 = sum_hm.astype(np.float32)
                    # minmax normalization
                    sum_display = _minmax_normalize(sum_hm)
                    sum_display = sum_display.astype(np.float32)
                    saliency_mask = get_saliency_mask(
                        sum_hm_f32,
                        percentile=float(args.bbox_percentile),
                        min_peak_fraction=float(args.bbox_min_peak_fraction),
                        open_kernel_size=ok,
                    )

                # plot per layer similarity maps
                plot_similarity_maps(
                    heatmaps,
                    cv2_img_bgr,
                    os.path.join(out_dir, f"layers_{location.id}.png"),
                    titles=titles,
                    suptitle=f"satclip per-layer — idx {n}",
                    ncols=4,
                    heatmap_normalize=norm,
                )
    
                # plot summed similarity map
                plot_similarity_maps(
                    [sum_display],
                    cv2_img_bgr,
                    os.path.join(out_dir, f"sum_layers_{location.id}.png"),
                    titles=[""],
                    suptitle="",
                    ncols=1,
                    figsize=(8.0, 8.0),
                    heatmap_normalize=norm,
                    pad_inches=0.0,
                )
                # plot bbox and mask
                if args.bbox and sum_hm_f32 is not None:
                    boxes, eroded_mask = boxes_from_saliency(
                        saliency_mask,
                        max_area=int(args.bbox_max_area),
                        img_h=h,
                        img_w=w,
                        square_side=int(args.bbox_square_side) if args.bbox_square_side is not None else None,
                    )

                    boxed_rgb = draw_boxes_on_rgb(
                        rgb_img,
                        boxes,
                        thickness=int(args.bbox_line_thickness),
                    )
                    cv2.imwrite(
                        os.path.join(out_dir, "bbox.png"),
                        cv2.cvtColor(boxed_rgb, cv2.COLOR_RGB2BGR),
                    )
                    # Binary foreground mask after threshold + opening (0 / 255).
                    cv2.imwrite(os.path.join(out_dir, "mask.png"), eroded_mask)
                    cv2.imwrite(os.path.join(out_dir, "original_mask.png"), saliency_mask)
                    # Crop each bbox from RGB and save under <out_dir>/bboxes/.
                    crops_dir = os.path.join(out_dir, "bboxes")
                    
                    if os.path.exists(crops_dir):
                        shutil.rmtree(crops_dir) # delete last results

                    os.makedirs(crops_dir)
                    sorted_boxes = sorted(
                        enumerate(boxes), key=lambda kv: kv[1][4], reverse=True
                    )
                    for rank, (_, (bx, by, bw, bh, area)) in enumerate(sorted_boxes):
                        x1 = max(0, int(bx))
                        y1 = max(0, int(by))
                        x2 = min(int(w), x1 + int(bw))
                        y2 = min(int(h), y1 + int(bh))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop_rgb = rgb_img[y1:y2, x1:x2]
                        crop_name = f"bbox_{rank:03d}_area{int(area)}_x{x1}_y{y1}_w{x2 - x1}_h{y2 - y1}.png"
                        cv2.imwrite(
                            os.path.join(crops_dir, crop_name),
                            cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                        )

if __name__ == "__main__":
    main()

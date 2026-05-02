"""SatCLIP CLIP-Surgery inference (Sentinel-2 GeoTIFFs)."""

from __future__ import annotations

import os
import sys
import shutil
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import clip
from images.clip_surgery import get_satclip
from images.clip_surgery.inference_utils import (
    RANDOM_LOCATION,
    encode_locations,
    load_image_sentinel,
    load_rgb_image,
    load_s2_sample,
    plot_layer_similarity_grid,
    safe_results_subdir,
    save_surgery_maps_to_png,
    surgery_similarity_patch_rows,
)
SUPPORTED_MODELS = {
    "vit16": "microsoft/SatCLIP-ViT16-L40",
    "resnet50": "microsoft/SatCLIP-ResNet50-L40",
}


def _centered_square_box(
    cx: float,
    cy: float,
    side: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Fixed-size square centered on (cx, cy), clipped to image bounds."""
    side = max(1, int(side))
    side = min(side, img_w, img_h)
    half = side // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, img_w - side))
    y0 = max(0, min(y0, img_h - side))
    return x0, y0, side, side

def _split_mask_by_erosion(binary_mask, max_area):
    final_output = np.zeros_like(binary_mask)
    kernel = np.ones((3,3), np.uint8)
    
    # Initial pass to get all blobs
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
    
    # Put blobs that are too big into a "todo" list; keep the rest
    todo_stack = []
    for i in range(1, num):
        blob_mask = (labels == i).astype(np.uint8) * 255
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            todo_stack.append(blob_mask)
        else:
            final_output = cv2.bitwise_or(final_output, blob_mask)

    # Process the large blobs until they are split or small enough
    while todo_stack:
        current_blob = todo_stack.pop()
        
        # Erode once to try and break it
        eroded = cv2.erode(current_blob, kernel, iterations=1)
        
        # Analyze the result of the erosion
        n_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(eroded)
        
        # If the blob vanished, stop (safety catch)
        if n_sub <= 1:
            continue

        for j in range(1, n_sub):
            child_mask = (sub_labels == j).astype(np.uint8) * 255
            child_area = sub_stats[j, cv2.CC_STAT_AREA]
            
            if child_area > max_area:
                # Still too big, put back on stack to erode again
                todo_stack.append(child_mask)
            else:
                # Finally small enough!
                final_output = cv2.bitwise_or(final_output, child_mask)
                
    return final_output

def get_saliency_mask(
    hm: np.ndarray,
    *,
    percentile: float,
    min_peak_fraction: float,
    open_kernel_size: int,
) -> np.ndarray:
    hm = np.asarray(hm, dtype=np.float64)
    hm_min = float(hm.min())
    hm_max = float(hm.max())
    if hm_max - hm_min < 1e-12:
        return np.zeros(hm.shape, dtype=np.uint8)

    p = float(np.clip(percentile, 0.0, 1.0))
    q_thr = float(np.quantile(hm, p))
    peak_thr = hm_max * float(np.clip(min_peak_fraction, 0.0, 1.0))
    thr = max(q_thr, peak_thr)
    mask = (hm >= thr).astype(np.uint8)

    k = int(open_kernel_size)
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def _boxes_from_saliency(
    original_mask: np.ndarray,
    *,
    max_area: int,
    img_h: int,
    img_w: int,
    square_side: int,
) -> tuple[list[tuple[int, int, int, int, int]], np.ndarray]:
    """Strong saliency only: threshold from raw summed heatmap (not min–max display).

    Uses ``thr = max(quantile(hm, percentile), min_peak_fraction * max(hm))`` so weak,
    stretched background from per-image normalization is not used for detection.
    """
    mask = original_mask.copy()
    mask = _split_mask_by_erosion(mask, max_area)

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    boxes: list[tuple[int, int, int, int, int]] = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        if square_side is not None:
            bx, by, bw, bh = _centered_square_box(cx, cy, square_side, img_w, img_h)
        else:
            bx, by, bw, bh = x, y, w, h
        boxes.append((bx, by, bw, bh, int(area)))
    return boxes, mask


def _draw_boxes_on_rgb(
    rgb_img: np.ndarray,
    boxes: list[tuple[int, int, int, int, int]],
    *,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on RGB image and return RGB result."""
    out_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for x, y, w, h, _ in boxes:
        cv2.rectangle(out_bgr, (x, y), (x + w, y + h), color_bgr, int(thickness))
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _load_rgb_from_images_corr(place: str, id_val: str) -> np.ndarray | None:
    """Load RGB preview PNG from images_corr/ (or Images_corr/) if present."""
    stem = Path(id_val).stem
    candidates = [
        Path(place) / "images_corr" / f"{stem}.png",
        Path(place) / "images" / f"{stem}.png",
    ]
    for p in candidates:
        if p.is_file():
            try:
                rgb = load_rgb_image(str(p))
            except Exception:
                continue
            return rgb
    return None


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="SatCLIP CLIP surgery maps for Sentinel-2 tiles.")
    parser.add_argument("--arquitecture", choices=["vit16", "resnet50"], default="vit16")
    
    parser.add_argument("--source_folder", type=str, default="/home/seri6958/Places")
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
    parser.add_argument(
        "--store_salient_embeddings",
        action="store_true",
        help="Store the embeddings of an image after applying the surgery mask",
    )
    parser.add_argument(
        "--no_plots_save",
        action="store_true",
        default=False,
        help="Do not save the plots and the surgery maps",
    )
    args = parser.parse_args()
    if args.bbox_kernel_size is not None:
        args.bbox_open_kernel = int(args.bbox_kernel_size)

    ok = int(args.bbox_open_kernel)
    if ok > 0 and ok % 2 == 0:
        ok += 1

    use_surgery = not args.no_surgery
    layer_grid = not args.no_layer_grid and use_surgery
    layer_grid_norm = args.layer_grid_normalize

    if args.arquitecture == "resnet50":
        raise ValueError("Resnet50 not implemented yet")

    satclip_model_path = SUPPORTED_MODELS[args.arquitecture]
    model = get_satclip(
        hf_hub_download(satclip_model_path, "satclip-vit16-l40.ckpt"),
        device=device,
        surgery=use_surgery,
        return_all=True,
    )
    model.eval()
    encode_location = model.encode_location

    places_folder = args.source_folder
    data_folder = "data"

    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
    ]

    for place in places:
        
        index_path = os.path.join(place, "index.csv")
        all_locations = load_s2_sample(index_path, "id")
        all_location_features = encode_locations(
            all_locations, encode_location, device, satclip=True
        )
        id_to_idx = {loc[0]: idx for idx, loc in enumerate(all_locations)}
        redundant_feats = encode_locations([RANDOM_LOCATION], encode_location, device, satclip=True)

        place_name = Path(place).name
        results_dir = _repo_root / "out_new" / "satclip" / place_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Computing satclip surgery maps for {place}...")
        similarity_maps_dict = {}
        rgb_images_dict = {}
        cv2_img_bgr_dict = {}

        for id_val, lon, lat in tqdm(all_locations, desc=f"CLIP Surgery {place}"):
            stem = Path(id_val).stem
            n = id_to_idx.get(id_val)
            if n is None:
                raise ValueError(f"Location {id_val} not found in {all_locations}")
            data_path = os.path.join(place, data_folder, id_val)
            try:
                image, rgb_img = load_image_sentinel(data_path, device=device)
                rgb_corr = _load_rgb_from_images_corr(place, id_val)
                if rgb_corr is not None:
                    rgb_img = rgb_corr
            except Exception as e:
                print(f"Error loading image {data_path}: {e}")
                continue

            out_dir = os.path.join(str(results_dir), safe_results_subdir(id_val))
            os.makedirs(out_dir, exist_ok=True)

            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                vm = model.visual
                if layer_grid and hasattr(vm, "forward_intermediates"):
                    final_tokens, layer_hiddens = vm.forward_intermediates(
                        image,
                        indices=None,
                        norm=False,
                        output_fmt="NLC",
                    )
                    image_features = vm.forward_head(final_tokens, pre_logits=False)
                else:
                    layer_hiddens = None
                    image_features = model.encode_image(image)

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

                if args.no_plots_save:
                    plot_layer_similarity_grid(
                        heatmaps,
                        cv2_img_bgr,
                        os.path.join(out_dir, f"layers_{stem}.png"),
                        titles=titles,
                        suptitle=f"satclip per-layer — idx {n}",
                        ncols=4,
                        heatmap_normalize=layer_grid_norm,
                    )
                if sum_display is not None and not args.no_plots_save:
                    plot_layer_similarity_grid(
                        [sum_display],
                        cv2_img_bgr,
                        os.path.join(out_dir, f"sum_layers_{stem}.png"),
                        titles=[""],
                        suptitle="",
                        ncols=1,
                        figsize=(8.0, 8.0),
                        heatmap_normalize=layer_grid_norm,
                        pad_inches=0.0,
                    )
                    if args.bbox and sum_hm_f32 is not None:
                        boxes, eroded_mask = _boxes_from_saliency(
                            saliency_mask,
                            max_area=int(args.bbox_max_area),
                            img_h=h,
                            img_w=w,
                            square_side=int(args.bbox_square_side) if args.bbox_square_side is not None else None,
                        )

                        boxed_rgb = _draw_boxes_on_rgb(
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
                            shutil.rmtree(crops_dir)  # Deletes the folder and all contents

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
        if not args.no_plots_save:
            save_surgery_maps_to_png(place, str(results_dir), similarity_maps_dict, cv2_img_bgr_dict)


if __name__ == "__main__":
    main()

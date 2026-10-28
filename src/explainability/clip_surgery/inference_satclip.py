"""SatCLIP layer maps with two location-mean nulls.

One null is the mean of 10,000 area-uniform locations. The other is the mean
of 10,000 locations drawn from SatCLIP's S2-100k training index.
"""

import csv
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from CLIP_Surgery import clip
from huggingface_hub import hf_hub_download
from inference_utils import (
    Location,
    boxes_from_saliency,
    centroid_null,
    correlation,
    draw_boxes_on_rgb,
    encode_locations,
    get_saliency_mask,
    load_image_sentinel,
    load_locations,
    load_rgb_from_images_corr,
    plot_similarity_maps,
    seed_everything,
)
from satclip_surgery.load import get_satclip
from tqdm import tqdm

PLACES_FOLDER = "/data/cities50"
S2_INDEX = Path("/data/s2_100k/index.csv")
SEED = 42
N_REF = 10_000
_KINDS = ("none", "random_earth", "train_dist")


def _satclip_patch_image_features(vm, image: torch.Tensor) -> torch.Tensor:
    """Per-token image embeddings from SatCLIP's VisionTransformer."""
    tokens, _ = vm.forward_intermediates(
        image,
        indices=None,
        norm=False,
        output_fmt="NLC",
    )
    if hasattr(vm, "norm"):
        tokens = vm.norm(tokens)
    elif hasattr(vm, "fc_norm"):
        tokens = vm.fc_norm(tokens)
    return vm.head(tokens)


def _sample_random_locations(n: int, rng: np.random.Generator) -> list[Location]:
    """Area-uniform sample on the sphere, so the poles are not over-weighted."""
    lon = rng.uniform(-180.0, 180.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return [
        Location(id=f"rand_{i}", lon=float(lon[i]), lat=float(lat[i]))
        for i in range(n)
    ]


def _sample_train_locations(n: int, rng: np.random.Generator) -> list[Location]:
    """Draw locations from SatCLIP's S2-100k training index."""
    with S2_INDEX.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    chosen = rng.choice(len(rows), size=n, replace=False)
    return [
        Location(
            id=str(rows[i]["fn"]),
            lon=float(rows[i]["lon"]),
            lat=float(rows[i]["lat"]),
        )
        for i in chosen
    ]


def _encode_reference(
    locations: list[Location], encode_fn, device: str
) -> torch.Tensor:
    parts = []
    for start in range(0, len(locations), 2048):
        with torch.no_grad():
            parts.append(
                encode_locations(
                    locations[start : start + 2048],
                    encode_fn,
                    device,
                    satclip=True,
                )
            )
    return torch.cat(parts, dim=0)


def _score_feat(
    feat,
    *,
    target_feat,
    query_index,
    hw,
    use_surgery,
    ref_feats,
    noise,
):
    """Score one projected hidden state against the query location."""
    feat = feat / feat.norm(dim=-1, keepdim=True)
    # standard similarity
    logits = feat @ target_feat.T
    sims = {"none": logits}
    # original code -> verified leads to same results as our new code
    # similarity = clip.clip_feature_surgery(
    #     image_features, target_feat, redundant_feats=redundant_feats
    # )
    # here we're going to subract the mean noise of the reference set from the logits
    # original code assumed reference set = target and you could weight
    # the logits by the weight of the target location.
    # here, our reference or null set is different.
    # The query is not in the reference set, so its logit stays unscaled.
    # A later pass could scale it by the weight of the most similar
    # reference location.
    if use_surgery and noise != "none":
        for kind in _KINDS[1:]:
            # weighted uses this output's class token
            mu, _ = centroid_null(
                ref_feats[kind], feat, weighted=(noise == "weighted")
            )
            sims[kind] = logits - feat @ mu.reshape(1, -1, 1)
    h, w = hw
    maps = {}
    for kind, similarity in sims.items():
        visual_patches_sim = similarity[:, 1:, :]  # strip CLS token
        similarity_map = clip.get_similarity_map(visual_patches_sim, (h, w))
        maps[kind] = similarity_map[0, :, :, query_index].cpu().numpy()
    return sims, maps


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(
        description="SatCLIP CLIP surgery maps for Sentinel-2 tiles."
    )
    parser.add_argument("--source_folder", type=str, default=PLACES_FOLDER)
    parser.add_argument("--lon_column", type=str, default="lon")
    parser.add_argument("--lat_column", type=str, default="lat")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--out_folder", type=str, default="out/satclip_null_compare")
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
    parser.add_argument(
        "--noise",
        choices=["mean", "weighted", "none"],
        default="mean",
        help="Noise subtracted from the location logits as done in CLIP Surgery. "
        "'mean' is the plain mean of each reference set, "
        "'weighted' is the class-token weighted mean, "
        "'none' leaves the logits unchanged.",
    )
    parser.add_argument(
        "--place",
        type=str,
        default=None,
        help="Run only this place folder under --source_folder, helpful for debugging.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Stop after this many images in each place, helpful for debugging.",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Detect salient regions on each mean-null layer sum.",
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
    kinds = ("none",) if args.noise == "none" else _KINDS

    model = get_satclip(
        hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
        device=device,
        surgery=use_surgery,
        return_all=True,
    )
    model.eval()
    encode_location = model.encode_location

    ref_feats = {}
    if use_surgery and args.noise != "none":
        rng = np.random.default_rng(SEED)
        ref_feats = {
            "random_earth": _encode_reference(
                _sample_random_locations(N_REF, rng), encode_location, device
            ),
            "train_dist": _encode_reference(
                _sample_train_locations(N_REF, rng), encode_location, device
            ),
        }
        mu_earth = ref_feats["random_earth"].mean(dim=0)
        mu_train = ref_feats["train_dist"].mean(dim=0)
        earth_hat = mu_earth / mu_earth.norm()
        train_hat = mu_train / mu_train.norm()
        print(
            f"10k means  noise={args.noise}  "
            f"||random_earth||={mu_earth.norm().item():.3f}  "
            f"||train_dist||={mu_train.norm().item():.3f}  "
            f"cosine={float((earth_hat @ train_hat).item()):+.3f}"
        )

    places_folder = args.source_folder
    places = sorted(
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
        and (args.place is None or place == args.place)
    )
    rows = []

    for place in places:
        index_path = os.path.join(place, "index.csv")
        if not os.path.isfile(index_path):
            continue
        all_locations_place = load_locations(
            index_path,
            lon_column=args.lon_column,
            lat_column=args.lat_column,
            id_column=args.id_column,
        )
        place_name = Path(place).name
        print(f"Computing satclip surgery maps for {place_name}...")
        vm = model.visual
        locations_to_run = all_locations_place
        if args.max_images is not None:
            locations_to_run = all_locations_place[: args.max_images]

        for location in tqdm(locations_to_run, desc=f"CLIP Surgery {place_name}"):
            data_path = os.path.join(place, "data", location.id)
            if not os.path.isfile(data_path):
                print(f"missing {place_name} {location.id}")
                continue
            rgb_img = load_rgb_from_images_corr(place, location.id)
            if rgb_img is None:
                print(f"missing rgb {place_name} {location.id}")
                continue
            target_feat = encode_locations(
                [location], encode_location, device, satclip=True
            )
            query_index = 0
            image, _ = load_image_sentinel(data_path, device=device)

            out_dir = os.path.join(args.out_folder, place_name, location.id)
            os.makedirs(out_dir, exist_ok=True)
            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                if layer_grid:
                    _, layer_hiddens = vm.forward_intermediates(
                        image,
                        indices=None,
                        norm=False,
                        output_fmt="NLC",
                    )
                    hidden_slice = layer_hiddens[-6:]
                    chosen = [
                        vm.forward_head(vm.norm(hid), pre_logits=False)
                        for hid in hidden_slice
                    ]
                    n_layers = len(layer_hiddens)
                elif use_surgery:
                    final_tokens, _ = vm.forward_intermediates(
                        image,
                        indices=None,
                        norm=False,
                        output_fmt="NLC",
                    )
                    chosen = [vm.forward_head(final_tokens, pre_logits=False)]
                    n_layers = 1
                else:
                    # All tokens, not only the class token.
                    chosen = [_satclip_patch_image_features(vm, image)]
                    n_layers = 1
                scored = []
                titles = []
                for li, feat in enumerate(chosen):
                    scored.append(
                        _score_feat(
                            feat,
                            target_feat=target_feat,
                            query_index=query_index,
                            hw=(h, w),
                            use_surgery=use_surgery,
                            ref_feats=ref_feats,
                            noise=args.noise,
                        )
                    )
                    if layer_grid:
                        titles.append(f"L{n_layers - min(6, n_layers) + li}")

            sum_maps = None
            if not layer_grid:
                similarities, query_maps = scored[0]
                if use_surgery and args.noise != "none":
                    none_vec = similarities["none"][0, 1:, query_index].reshape(-1)
                    earth_vec = similarities["random_earth"][
                        0, 1:, query_index
                    ].reshape(-1)
                    train_vec = similarities["train_dist"][
                        0, 1:, query_index
                    ].reshape(-1)
                    row = {
                        "place": place_name,
                        "id": str(location.id),
                        "final_random_earth": correlation(
                            none_vec.cpu(), earth_vec.cpu()
                        ),
                        "final_train_dist": correlation(
                            none_vec.cpu(), train_vec.cpu()
                        ),
                        "final_random_earth_train_dist": correlation(
                            earth_vec.cpu(), train_vec.cpu()
                        ),
                    }
                    rows.append(row)
                    print(
                        f"  final random_earth={row['final_random_earth']:+.3f}  "
                        f"train_dist={row['final_train_dist']:+.3f}  "
                        f"random_earth_vs_train_dist="
                        f"{row['final_random_earth_train_dist']:+.3f}"
                    )
                plot_similarity_maps(
                    [query_maps[kind] for kind in kinds],
                    cv2_img_bgr,
                    os.path.join(out_dir, "compare.png"),
                    titles=["no null", "random_earth", "train_dist"][: len(kinds)],
                    suptitle=f"{place_name} {location.id}",
                    ncols=len(kinds),
                    heatmap_normalize=norm,
                )
            else:
                per_kind = {kind: [] for kind in kinds}
                for _, layer_maps in scored:
                    for kind in kinds:
                        per_kind[kind].append(layer_maps[kind])
                sum_maps = {}
                for kind in kinds:
                    heatmaps = per_kind[kind]
                    sum_hm = np.zeros_like(heatmaps[0], dtype=np.float64)
                    for hm in heatmaps:
                        sum_hm = sum_hm + hm
                    smin, smax = float(sum_hm.min()), float(sum_hm.max())
                    sum_display = ((sum_hm - smin) / (smax - smin + 1e-8)).astype(
                        np.float32
                    )
                    sum_maps[kind] = sum_hm
                    plot_similarity_maps(
                        heatmaps,
                        cv2_img_bgr,
                        os.path.join(out_dir, f"{kind}_layers.png"),
                        titles=titles,
                        suptitle=f"{kind} — {location.id}",
                        ncols=4,
                        heatmap_normalize=norm,
                    )
                    plot_similarity_maps(
                        [sum_display],
                        cv2_img_bgr,
                        os.path.join(out_dir, f"{kind}_sum.png"),
                        titles=[""],
                        suptitle="",
                        ncols=1,
                        figsize=(8.0, 8.0),
                        heatmap_normalize=norm,
                        pad_inches=0.0,
                    )
                if args.noise != "none":
                    row = {
                        "place": place_name,
                        "id": str(location.id),
                        "sum_random_earth": correlation(
                            sum_maps["none"], sum_maps["random_earth"]
                        ),
                        "sum_train_dist": correlation(
                            sum_maps["none"], sum_maps["train_dist"]
                        ),
                        "sum_random_earth_train_dist": correlation(
                            sum_maps["random_earth"], sum_maps["train_dist"]
                        ),
                    }
                    rows.append(row)
                    print(
                        f"  sum   random_earth={row['sum_random_earth']:+.3f}  "
                        f"train_dist={row['sum_train_dist']:+.3f}  "
                        f"random_earth_vs_train_dist="
                        f"{row['sum_random_earth_train_dist']:+.3f}"
                    )
                    plot_similarity_maps(
                        [sum_maps[kind] for kind in kinds],
                        cv2_img_bgr,
                        os.path.join(out_dir, "compare_sum.png"),
                        titles=["no null", "random_earth", "train_dist"],
                        suptitle=f"{place_name} {location.id}",
                        ncols=3,
                        heatmap_normalize=norm,
                    )

            if args.bbox and sum_maps is not None:
                for kind in kinds:
                    if kind == "none":
                        continue
                    sum_hm_f32 = sum_maps[kind].astype(np.float32)
                    saliency_mask = get_saliency_mask(
                        sum_hm_f32,
                        percentile=float(args.bbox_percentile),
                        min_peak_fraction=float(args.bbox_min_peak_fraction),
                        open_kernel_size=ok,
                    )
                    boxes, eroded_mask = boxes_from_saliency(
                        saliency_mask,
                        max_area=int(args.bbox_max_area),
                        img_h=h,
                        img_w=w,
                        square_side=int(args.bbox_square_side)
                        if args.bbox_square_side is not None
                        else None,
                    )
                    boxed_rgb = draw_boxes_on_rgb(
                        rgb_img,
                        boxes,
                        thickness=int(args.bbox_line_thickness),
                    )
                    cv2.imwrite(
                        os.path.join(out_dir, f"{kind}_bbox.png"),
                        cv2.cvtColor(boxed_rgb, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(os.path.join(out_dir, f"{kind}_mask.png"), eroded_mask)
                    cv2.imwrite(
                        os.path.join(out_dir, f"{kind}_original_mask.png"),
                        saliency_mask,
                    )
                    crops_dir = os.path.join(out_dir, f"{kind}_bboxes")
                    if os.path.exists(crops_dir):
                        shutil.rmtree(crops_dir)
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
                        crop_name = (
                            f"bbox_{rank:03d}_area{int(area)}_x{x1}_y{y1}_"
                            f"w{x2 - x1}_h{y2 - y1}.png"
                        )
                        cv2.imwrite(
                            os.path.join(crops_dir, crop_name),
                            cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                        )

    _write_correlation_report(Path(args.out_folder) / "map_correlations.txt", rows)


def _write_correlation_report(path: Path, rows: list[dict]) -> None:
    """Write each image and its map correlations."""
    keys = (
        "final_random_earth",
        "final_train_dist",
        "final_random_earth_train_dist",
        "sum_random_earth",
        "sum_train_dist",
        "sum_random_earth_train_dist",
    )
    header = (
        f"{'place':<16} {'id':<55} "
        f"{'final_random_earth':>18} {'final_train_dist':>16} "
        f"{'final_random_earth_train_dist':>30} "
        f"{'sum_random_earth':>16} {'sum_train_dist':>14} "
        f"{'sum_random_earth_train_dist':>28}"
    )
    lines = [header]
    for row in sorted(rows, key=lambda item: (item["place"], item["id"])):
        lines.append(
            f"{row['place']:<16} {row['id']:<55} "
            f"{row.get('final_random_earth', float('nan')):+18.3f} "
            f"{row.get('final_train_dist', float('nan')):+16.3f} "
            f"{row.get('final_random_earth_train_dist', float('nan')):+30.3f} "
            f"{row.get('sum_random_earth', float('nan')):+16.3f} "
            f"{row.get('sum_train_dist', float('nan')):+14.3f} "
            f"{row.get('sum_random_earth_train_dist', float('nan')):+28.3f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["place", "id", *keys])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

"""GeoCLIP layer maps with two location-mean nulls.

One null is the mean of 10,000 area-uniform locations. The other is the mean
of 10,000 locations drawn from GeoCLIP's 100k photo-location gallery.
"""

import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import geoclip
import numpy as np
import torch
from CLIP_Surgery import clip
from geoclip_surgery import get_geoclip
from inference_utils import (
    Location,
    centroid_null,
    correlation,
    encode_locations,
    load_image_geoclip,
    load_locations,
    plot_similarity_maps,
    seed_everything,
)
from location_centroid import _sample_random_locations
from PIL import Image
from tqdm import tqdm

IMAGES_FOLDER = "/data/im2gps300/Places_from_im2gps"
GALLERY_CSV = (
    Path(geoclip.__file__).resolve().parent
    / "model"
    / "gps_gallery"
    / "coordinates_100K.csv"
)
SEED = 42
N_REF = 10_000
_THIS_DIR = Path(__file__).resolve().parent
_KINDS = ("none", "random_earth", "train_dist")



def _sample_gallery_locations(n: int, rng: np.random.Generator) -> list[Location]:
    """Draw locations from GeoCLIP's photo-location gallery."""
    with GALLERY_CSV.open(newline="") as handle:
        gallery = list(csv.DictReader(handle))
    chosen = rng.choice(len(gallery), size=n, replace=False)
    return [
        Location(
            id=f"gallery_{i}",
            lon=float(gallery[i]["LON"]),
            lat=float(gallery[i]["LAT"]),
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
                    satclip=False,
                )
            )
    return torch.cat(parts, dim=0)


def _score_hidden(
    hidden,
    *,
    vm,
    vproj,
    mlp,
    target_feat,
    query_index,
    hw,
    use_surgery,
    ref_feats,
    noise,
):
    """Project one hidden state and score it against the query location."""
    feat = mlp(vproj(vm.post_layernorm(hidden)))
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
    parser = ArgumentParser(description="GeoCLIP CLIP surgery maps for RGB images.")
    parser.add_argument("--lon_column", type=str, default="lon")
    parser.add_argument("--lat_column", type=str, default="lat")
    parser.add_argument("--id_column", type=str, default="id")
    parser.add_argument("--source_folder", type=str, default=IMAGES_FOLDER)
    parser.add_argument("--out_folder", type=str, default="out")
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
    args = parser.parse_args()

    seed_everything(SEED)
    use_surgery = not args.no_surgery
    layer_grid = not args.no_layer_grid and use_surgery
    norm = args.normalize
    kinds = ("none",) if args.noise == "none" else _KINDS

    geo_model = get_geoclip(device=device, surgery=use_surgery, return_all=True)
    geo_model.eval()
    encode_location = geo_model.location_encoder.forward

    ref_feats = {}
    if use_surgery and args.noise != "none":
        rng = np.random.default_rng(SEED)
        ref_feats = {
            "random_earth": _encode_reference(
                _sample_random_locations(N_REF, rng), encode_location, device
            ),
            "train_dist": _encode_reference(
                _sample_gallery_locations(N_REF, rng), encode_location, device
            ),
        }
        mu_earth = ref_feats["random_earth"].mean(dim=0)
        mu_gallery = ref_feats["train_dist"].mean(dim=0)
        earth_hat = mu_earth / mu_earth.norm()
        gallery_hat = mu_gallery / mu_gallery.norm()
        print(
            f"10k means  noise={args.noise}  "
            f"||random_earth||={mu_earth.norm().item():.3f}  "
            f"||train_dist||={mu_gallery.norm().item():.3f}  "
            f"cosine={float((earth_hat @ gallery_hat).item()):+.3f}"
        )
    places_folder = args.source_folder

    places = [
        os.path.join(places_folder, place)
        for place in os.listdir(places_folder)
        if os.path.isdir(os.path.join(places_folder, place))
        and (args.place is None or place == args.place)
    ]
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
        run_name = "geoclip_no_surgery" if not use_surgery else "geoclip_surgery"
        results_dir = _THIS_DIR / "out" / run_name / place_name
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"Computing geoclip surgery maps for {place_name}...")

        vm = geo_model.image_encoder.CLIP.vision_model
        vproj = geo_model.image_encoder.CLIP.visual_projection
        mlp = geo_model.image_encoder.mlp

        locations_to_run = all_locations_place
        if args.max_images is not None:
            locations_to_run = all_locations_place[: args.max_images]

        for location in tqdm(locations_to_run, desc=f"CLIP Surgery {place_name}"):
            data_path = os.path.join(place, "images", str(location.id))
            if not os.path.isfile(data_path):
                print(f"missing {place_name} {location.id}")
                continue
            target_feat = encode_locations(
                [location], encode_location, device, satclip=False
            )
            query_index = 0

            image = load_image_geoclip(data_path, device=device)
            rgb_img = np.array(Image.open(data_path).convert("RGB"))

            out_dir = os.path.join(args.out_folder, place_name, location.id)
            os.makedirs(out_dir, exist_ok=True)

            cv2_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            h, w = rgb_img.shape[:2]

            with torch.no_grad():
                if layer_grid:
                    _, layer_hiddens = vm.forward_intermediates(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                    )
                    chosen = layer_hiddens[-6:]
                else:
                    vision_out = vm(
                        pixel_values=image,
                        interpolate_pos_encoding=False,
                    )
                    chosen = [vision_out.last_hidden_state]
                scored = []
                titles = []
                for li, hidden in enumerate(chosen):
                    scored.append(
                        _score_hidden(
                            hidden,
                            vm=vm,
                            vproj=vproj,
                            mlp=mlp,
                            target_feat=target_feat,
                            query_index=query_index,
                            hw=(h, w),
                            use_surgery=use_surgery,
                            ref_feats=ref_feats,
                            noise=args.noise,
                        )
                    )
                    if layer_grid:
                        titles.append(
                            f"L{len(layer_hiddens) - min(6, len(layer_hiddens)) + li}"
                        )

            if not layer_grid:
                similarities, query_maps = scored[0]
                if use_surgery and args.noise != "none":
                    none_vec = similarities["none"][0, 1:, query_index].reshape(-1)
                    earth_vec = similarities["random_earth"][
                        0, 1:, query_index
                    ].reshape(-1)
                    gallery_vec = similarities["train_dist"][
                        0, 1:, query_index
                    ].reshape(-1)
                    row = {
                        "place": place_name,
                        "id": str(location.id),
                        "final_random_earth": correlation(
                            none_vec.cpu(), earth_vec.cpu()
                        ),
                        "final_train_dist": correlation(
                            none_vec.cpu(), gallery_vec.cpu()
                        ),
                        "final_random_earth_train_dist": correlation(
                            earth_vec.cpu(), gallery_vec.cpu()
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

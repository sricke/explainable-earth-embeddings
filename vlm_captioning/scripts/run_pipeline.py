#!/usr/bin/env python3
"""
End-to-end satellite captioning: STAC download, optional preprocess, VLM inference, CSV export.

Run from the ``vlm_captioning`` directory (or ensure it is on ``PYTHONPATH``):

  python scripts/run_pipeline.py download --bbox -122.5 37.7 -122.4 37.8 \\
      --datetime 2024-06-01T00:00:00Z/2024-09-01T00:00:00Z --out data/tile.png

  python scripts/run_pipeline.py caption --manifest data/manifest.csv --out outputs/captions.csv

  python scripts/run_pipeline.py full --bbox ... --datetime ... --out-dir runs/demo \\
      --center-lon -122.45 --center-lat 37.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/run_pipeline.py` from repo or package root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Download-only path imports only ``data.dataset`` (rasterio / requests). Heavy deps load per subcommand.
from data.dataset import TileSpec, download_tile

DEFAULT_MOLMO2_ID = "allenai/Molmo2-8B"


def _cmd_download(args: argparse.Namespace) -> None:
    spec_kw: dict = {
        "bbox": tuple(args.bbox),
        "datetime_range": args.datetime,
    }
    if args.collection is not None:
        spec_kw["collection"] = args.collection
    spec = TileSpec(**spec_kw)
    path = download_tile(
        spec,
        args.out,
        stac_url=args.stac_url,
        collection=args.collection,
        prefer_high_res=args.prefer_high_res,
    )
    print(path)


def _cmd_preprocess(args: argparse.Namespace) -> None:
    from data.preprocess import preprocess_one

    preprocess_one(
        args.input,
        args.out,
        crop_size=tuple(args.crop) if args.crop else None,
        center_crop=not args.no_center_crop,
        resize=tuple(args.resize) if args.resize else None,
        super_resolve=args.super_resolve,
        sr_scale=args.sr_scale,
    )
    print(args.out)


def _cmd_caption(args: argparse.Namespace) -> None:
    from inference.caption_generator import CaptionGenerator
    from models.vlm_model import VLMModel
    from utils.io import save_captions
    from utils.logging import setup_logger

    logger = setup_logger(log_file=args.log_file)
    import pandas as pd

    df = pd.read_csv(args.manifest)
    for c in ("lat", "lon", "image_path"):
        if c not in df.columns:
            raise SystemExit(f"Manifest must include column {c!r}; got {list(df.columns)}")

    logger.info("Loading VLM (%s)...", args.model)
    model = VLMModel(
        model_id=args.model,
        fallback_blip=not args.no_blip_fallback,
    )
    gen = CaptionGenerator(
        model,
        prompt_style=args.prompt_style,
        merge_prompts=not args.no_merge_prompts,
    )

    if args.super_resolve_load:
        from data.data_loader import SatelliteImageDataset

        ds = SatelliteImageDataset(
            df,
            super_resolve=True,
            sr_scale=args.sr_scale,
        )
        out_df = gen.caption_dataset(
            ds,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        out_df = gen.caption_dataframe(
            df,
            max_new_tokens=args.max_new_tokens,
            show_progress=True,
        )

    save_captions(out_df, args.out)
    logger.info("Wrote %s (%d rows)", args.out, len(out_df))
    print(args.out)


def _cmd_full(args: argparse.Namespace) -> None:
    """Download one tile, optionally preprocess, caption, write CSV."""
    from data.preprocess import preprocess_one
    from inference.caption_generator import CaptionGenerator
    from models.vlm_model import VLMModel
    from utils.io import save_captions
    from utils.logging import setup_logger

    logger = setup_logger(log_file=args.log_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "tile_raw.png"
    proc_path = out_dir / "tile_preprocessed.png"

    spec_kw: dict = {
        "bbox": tuple(args.bbox),
        "datetime_range": args.datetime,
    }
    if args.collection is not None:
        spec_kw["collection"] = args.collection
    spec = TileSpec(**spec_kw)
    logger.info("Downloading tile...")
    download_tile(
        spec,
        raw_path,
        stac_url=args.stac_url,
        collection=args.collection,
        prefer_high_res=args.prefer_high_res,
    )

    image_for_vlm = raw_path
    if args.preprocess:
        logger.info("Preprocessing...")
        preprocess_one(
            raw_path,
            proc_path,
            crop_size=tuple(args.crop) if args.crop else None,
            center_crop=not args.no_center_crop,
            resize=tuple(args.resize) if args.resize else None,
            super_resolve=args.super_resolve,
            sr_scale=args.sr_scale,
        )
        image_for_vlm = proc_path

    logger.info("Loading VLM (%s)...", args.model)
    model = VLMModel(model_id=args.model, fallback_blip=not args.no_blip_fallback)
    gen = CaptionGenerator(model, prompt_style=args.prompt_style)
    caption = gen.caption_image(image_for_vlm, max_new_tokens=args.max_new_tokens)

    import pandas as pd

    row = {
        "lat": args.center_lat,
        "lon": args.center_lon,
        "image_path": str(image_for_vlm.resolve()),
        "caption": caption,
    }
    df = pd.DataFrame([row])
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "bbox": list(args.bbox),
                "datetime": args.datetime,
                "collection": spec.collection,
                "prefer_high_res": args.prefer_high_res,
                "raw_image": str(raw_path),
                "captioned_image": str(image_for_vlm),
            },
            indent=2,
        )
    )
    out_csv = out_dir / args.out_csv_name
    save_captions(df, out_csv)
    logger.info("Wrote %s", out_csv)
    print(caption)
    print(out_csv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Satellite VLM captioning pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Download a RGB tile via STAC (Sentinel-2 / NAIP)")
    d.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    d.add_argument("--datetime", required=True, help='ISO interval, e.g. "2024-06-01/2024-09-01"')
    d.add_argument("--out", type=Path, required=True, help="Output .png or .tif")
    d.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1")
    d.add_argument(
        "--collection",
        default=None,
        choices=("sentinel-2-l2a", "naip"),
        help="Force STAC collection (default: auto with --prefer-high-res)",
    )
    d.add_argument(
        "--prefer-high-res",
        action="store_true",
        help="Try NAIP (US, higher GSD) before Sentinel-2",
    )
    d.set_defaults(func=_cmd_download)

    pr = sub.add_parser("preprocess", help="Crop / resize / optional super-resolve one image")
    pr.add_argument("--input", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--crop", type=int, nargs=2, metavar=("W", "H"), default=None)
    pr.add_argument("--no-center-crop", action="store_true")
    pr.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    pr.add_argument("--super-resolve", action="store_true")
    pr.add_argument("--sr-scale", type=int, default=2)
    pr.set_defaults(func=_cmd_preprocess)

    c = sub.add_parser("caption", help="Caption images listed in a manifest CSV")
    c.add_argument("--manifest", type=Path, required=True, help="CSV with lat, lon, image_path")
    c.add_argument("--out", type=Path, required=True, help="Output captions CSV")
    c.add_argument("--model", default=DEFAULT_MOLMO2_ID)
    c.add_argument("--max-new-tokens", type=int, default=512)
    c.add_argument("--prompt-style", choices=("default", "detailed"), default="default")
    c.add_argument("--no-merge-prompts", action="store_true", help="Use first prompt only")
    c.add_argument("--no-blip-fallback", action="store_true", help="Fail if Molmo2 cannot load")
    c.add_argument("--batch-size", type=int, default=1)
    c.add_argument("--num-workers", type=int, default=0)
    c.add_argument("--super-resolve-load", action="store_true", help="Upscale each image while loading")
    c.add_argument("--sr-scale", type=int, default=2)
    c.add_argument("--log-file", default=None)
    c.set_defaults(func=_cmd_caption)

    f = sub.add_parser("full", help="Download bbox tile, optional preprocess, single caption + CSV")
    f.add_argument("--bbox", type=float, nargs=4, required=True)
    f.add_argument("--datetime", required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--center-lon", type=float, required=True, help="Logged lon for the output row")
    f.add_argument("--center-lat", type=float, required=True, help="Logged lat for the output row")
    f.add_argument("--out-csv-name", default="captions.csv")
    f.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1")
    f.add_argument("--collection", default=None, choices=("sentinel-2-l2a", "naip"))
    f.add_argument("--prefer-high-res", action="store_true")
    f.add_argument("--preprocess", action="store_true")
    f.add_argument("--crop", type=int, nargs=2, default=None)
    f.add_argument("--no-center-crop", action="store_true")
    f.add_argument("--resize", type=int, nargs=2, default=None)
    f.add_argument("--super-resolve", action="store_true")
    f.add_argument("--sr-scale", type=int, default=2)
    f.add_argument("--model", default=DEFAULT_MOLMO2_ID)
    f.add_argument("--max-new-tokens", type=int, default=512)
    f.add_argument("--prompt-style", choices=("default", "detailed"), default="default")
    f.add_argument("--no-blip-fallback", action="store_true")
    f.add_argument("--log-file", default=None)
    f.set_defaults(func=_cmd_full)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

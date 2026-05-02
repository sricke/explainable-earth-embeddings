#!/usr/bin/env python3
"""
Global 2–98% stretch on true-color (B04/B03/B02) from existing patch GeoTIFFs.

Scans ``**/data/*.tif`` under ``--input-root``, pools **all** reflectance values from
those bands across **every** file (shared single lo/hi for R, G, and B), then writes
PNG previews to a sibling ``images_corr/`` folder (same layout as ``data/``).

Example::

  export PYTHONPATH=src
  python scripts/postprocess_images_corr.py --input-root local_output/run/my-run

  # All runs under local_output (one global stretch for every patch found)
  python scripts/postprocess_images_corr.py --input-root local_output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import rasterio

# Match worker GeoTIFF nodata for float32 BOA rasters
_NODATA = -9999.0


def _find_data_tifs(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("data/*.tif")):
        if p.is_file():
            out.append(p)
    return out


def _read_truecolor_rgb(src: rasterio.io.DatasetReader) -> np.ndarray:
    """B04,B03,B02 → (H, W, 3) float32; nodata → nan."""
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    rgb = np.stack([b4, b3, b2], axis=-1)
    bad = ~np.isfinite(rgb) | (rgb == _NODATA)
    rgb = np.where(bad, np.nan, rgb)
    return rgb


def _gather_values_for_percentile(
    paths: list[Path],
    *,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Concatenate all R,G,B values (valid only), optionally subsample for memory."""
    chunks: list[np.ndarray] = []
    for p in paths:
        with rasterio.open(p) as src:
            rgb = _read_truecolor_rgb(src)
        flat = rgb.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            chunks.append(flat)
    if not chunks:
        return np.array([], dtype=np.float32)
    allv = np.concatenate(chunks, axis=0)
    if allv.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(allv.size, size=max_samples, replace=False)
        allv = allv[idx]
    return allv


def _rgb_to_uint8_stretch(
    rgb: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    valid = np.isfinite(rgb).all(axis=-1)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo_f = np.nanmin(rgb)
        hi_f = np.nanmax(rgb)
        if np.isfinite(lo_f) and np.isfinite(hi_f) and hi_f > lo_f:
            lo, hi = float(lo_f), float(hi_f)
        else:
            lo, hi = 0.0, 1.0
    out = (np.clip(np.nan_to_num(rgb, nan=lo), lo, hi) - lo) / (hi - lo) * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write images_corr/ PNGs with global 2–98% stretch from data/*.tif",
    )
    ap.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root to scan for **/data/*.tif (e.g. local_output/run/my-run or local_output)",
    )
    ap.add_argument(
        "--p-low",
        type=float,
        default=2.0,
        help="Lower percentile (default 2)",
    )
    ap.add_argument(
        "--p-high",
        type=float,
        default=98.0,
        help="Upper percentile (default 98)",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=5_000_000,
        help="Max values to use when estimating global percentiles (memory cap)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for subsampling (default 0)",
    )
    args = ap.parse_args()

    root = args.input_root.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    paths = _find_data_tifs(root)
    if not paths:
        print(f"No data/*.tif under {root}", file=sys.stderr)
        return 1

    print(f"Found {len(paths)} GeoTIFFs under {root}")
    vals = _gather_values_for_percentile(
        paths,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if vals.size == 0:
        print("No valid pixels in any file.", file=sys.stderr)
        return 1

    lo, hi = np.nanpercentile(vals, [args.p_low, args.p_high])
    print(f"Global percentiles ({args.p_low}–{args.p_high}%): lo={lo:.6g} hi={hi:.6g}")

    written = 0
    for p in paths:
        out_dir = p.parent.parent / "images_corr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{p.stem}.png"
        with rasterio.open(p) as src:
            rgb = _read_truecolor_rgb(src)
        u8 = _rgb_to_uint8_stretch(rgb, float(lo), float(hi))
        PIL.Image.fromarray(u8).save(out_png)
        written += 1

    print(f"Wrote {written} PNGs to **/images_corr/ next to each data/ folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

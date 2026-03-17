#!/usr/bin/env python
import argparse
from pathlib import Path

from splice.config import load_config
from splice.inference import main as splice_main


def _cli() -> None:
    """Backward-compatible CLI entry point for splice inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splice.yaml"),
        help="Path to splice config YAML.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    splice_main(cfg)


if __name__ == "__main__":
    _cli()
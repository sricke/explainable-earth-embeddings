from pathlib import Path
from typing import Any, Dict, List

import yaml


def _resolve_ckpt_path(root: Path, model_cfg: Dict[str, Any]) -> Path | None:
    """
    Resolve a checkpoint path, supporting either:
    - ckpt_path: full path or relative to project root, or with '~'
    - ckpt_dir + ckpt_name: dir (possibly with '~') plus filename
    """
    ckpt_path = model_cfg.get("ckpt_path")
    ckpt_dir = model_cfg.get("ckpt_dir")
    ckpt_name = model_cfg.get("ckpt_name")

    # Prefer explicit dir + name if provided
    if ckpt_dir is not None and ckpt_name is not None:
        base = Path(ckpt_dir).expanduser()
        if not base.is_absolute():
            base = (root / base).resolve()
        return (base / ckpt_name).resolve()

    if ckpt_path is None:
        return None

    p = Path(ckpt_path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def load_config(path: Path) -> Dict[str, Any]:
    """Load raw splice config YAML and resolve paths."""
    with path.open("r") as f:
        raw = yaml.safe_load(f)

    # Assume configs/splice.yaml -> project root one level up
    root = path.parent.parent

    # Resolve vocab sources
    vocab_files: List[Path] = []
    if "vocab_files" in raw:
        vocab_files = [(root / p).resolve() for p in raw["vocab_files"]]
    elif "vocab_dirs" in raw:
        for d in raw["vocab_dirs"]:
            vocab_files.extend(sorted((root / d).resolve().glob("*.json")))
    else:
        raise KeyError("Config must define either 'vocab_files' or 'vocab_dirs'.")

    # Resolve model checkpoint paths, supporting ckpt_dir/ckpt_name or ckpt_path
    models: list[dict[str, Any]] = []
    for m in raw["models"]:
        resolved = dict(m)
        resolved["ckpt_path"] = _resolve_ckpt_path(root, m)
        models.append(resolved)

    return {
        "device": raw["device"],
        "num_samples": raw.get("num_samples"),
        "batch_size": raw["batch_size"],
        "max_vocab_size": raw["max_vocab_size"],
        "dataset_csv": (root / raw["dataset_csv"]).resolve(),
        "models": models,
        "vocab_files": vocab_files,
        "l1_penalties": raw["l1_penalties"],
        "top_k": raw.get("top_k", 5),
        "known_locations": {k: tuple(v) for k, v in raw["known_locations"].items()},
        "parquet_dir": (root / raw["output"]["parquet_dir"]).resolve(),
        "report_dir": (root / raw["output"]["report_dir"]).resolve(),
    }


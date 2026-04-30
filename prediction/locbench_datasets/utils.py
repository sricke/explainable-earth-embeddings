from .fmow import FMoWDataset
from .birds import EbirdDataset
from .yfcc import YFCCDataset
from .mosaiks import MosaiksDataset
from .sustainbench import SustainBenchDataset


def _fmt(paths, split):
    return {k: v.format(split=split) if isinstance(v, str) else v for k, v in paths.items()}


def build_dataset(name, split, paths, **kwargs):
    p = _fmt(paths, split)
    if name == "fmow":
        return FMoWDataset(split, p["data"], p["annotation"], **kwargs)
    if name in ("nabirds", "birdsnap"):
        return EbirdDataset(split, p["meta"], **kwargs)
    if name == "yfcc":
        return YFCCDataset(split, p["meta"], **kwargs)
    if name == "mosaiks":
        return MosaiksDataset(split, p["csv"], **kwargs)
    if name == "sustainbench":
        return SustainBenchDataset(split, p["trainval"], p["test"], **kwargs)
    raise ValueError(f"Unknown dataset {name}")
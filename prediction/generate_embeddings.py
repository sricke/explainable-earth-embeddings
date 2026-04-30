import argparse
import sys
import yaml

sys.path.append("..")
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from locbench_datasets.base_dataset import dataset_kwargs
from locbench_datasets.utils import build_dataset
from splice import _LocWrapper

_PATHS_YAML = Path(__file__).parent / "paths.yaml"


def resolve_dataset_config(dataset: str, cfg: dict):
    root = cfg.get("root", "")
    ds, *rest = dataset.split(".", 1)
    spec = cfg.get(ds)
    if spec is None:
        raise ValueError(f"Unknown dataset: {ds}")
    if rest:
        spec = spec.get(rest[0])
        if spec is None:
            raise ValueError(f"Unknown {ds} subtype: {rest[0]}")
    spec = {k: v.replace("${root}", root) if isinstance(v, str) else v for k, v in spec.items()}
    return ds, spec, spec.pop("value_col")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["satclip", "geoclip", "sparse"], required=True)
    p.add_argument("--dataset", required=True, help="e.g. fmow, nabirds, mosaiks.elevation")
    p.add_argument("--config", default=str(_PATHS_YAML))
    p.add_argument("--root", default=None, help="Override root path from config")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--value_col", default=None, help="Override value_col from config")
    p.add_argument("--lat_col", default="lat")
    p.add_argument("--lon_col", default="lon")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model_path", default=None, help="[sparse] SpLiCE checkpoint")
    p.add_argument("--method", default="splice", choices=["splice", "sae", "bovw"])
    p.add_argument("--concepts_json", default=None)
    return p.parse_args()


def _load_model(args, device):
    if args.encoder == "sparse":
        if not args.model_path:
            raise ValueError("--model_path required for sparse encoder")
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        if not hasattr(model, "encode_image"):
            model = _LocWrapper(model)
    else:
        from models.location_encoder import load_model
        model = load_model(args.encoder, device=device)
    return model.to(device).eval()


def _encode(encoder, model, loc, device):
    with torch.no_grad():
        if encoder == "sparse":
            codes, _ = model.encode_image(loc.to(device))
            return codes.cpu()
        x = loc[:, [1, 0]].double().to(device) if encoder == "satclip" else loc.float().to(device)
        return F.normalize(model(x).float(), dim=-1).cpu()


def encode_split(args, model, split, device, name, entry, ds_kw):
    loader = DataLoader(build_dataset(name, split, entry, **ds_kw),
                        batch_size=args.batch_size, shuffle=False)
    codes, vals, locs = zip(*[
        (_encode(args.encoder, model, loc, device), val, loc)
        for loc, val in tqdm(loader, desc=f"Encoding {split}")
    ])
    return torch.cat(codes), torch.cat(vals), torch.cat(locs)


def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.root:
        cfg["root"] = args.root
    ds, entry, value_col = resolve_dataset_config(args.dataset, cfg)
    ds_kw = dict(value_col=args.value_col or value_col,
                 lat_col=args.lat_col, lon_col=args.lon_col,
                 **dataset_kwargs(args.dataset))

    model = _load_model(args, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.method if args.encoder == "sparse" else args.encoder

    for split in args.splits:
        codes, vals, locs = encode_split(args, model, split, device, ds, entry, ds_kw)
        torch.save({"codes": codes, "vals": vals, "locs": locs}, out_dir / f"{prefix}_{split}.pt")
    if args.concepts_json:
        (out_dir / "concepts.json").write_text(Path(args.concepts_json).read_text())


if __name__ == "__main__":
    main()

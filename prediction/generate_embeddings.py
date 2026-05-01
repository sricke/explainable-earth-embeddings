import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("..")

from locbench_datasets.base_dataset import dataset_kwargs
from locbench_datasets.utils import build_dataset
from prediction import load_paths
from splice import _LocWrapper


def get_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--encoder", choices=["satclip", "geoclip"])
    g.add_argument("--sparse_model", help="Named sparse model from paths.yaml (e.g. geoclip_geoyfcc)")
    p.add_argument("--dataset", required=True, help="e.g. fmow, nabirds, mosaiks.elevation")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_model(args, paths, device):
    if args.sparse_model:
        spec = paths["sparse_models"]["splice"][args.sparse_model] # hardcoded for splice for now
        model = torch.load(spec["model"], map_location=device, weights_only=False)
        model.solver = "admm"
        model.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(model, "encode_image"):
            model = _LocWrapper(model)
    else:
        from models.location_encoder import load_model
        model = load_model(args.encoder, device=device)
    return model.to(device).eval()


def _encode(args, model, loc, device):
    with torch.no_grad():
        if args.sparse_model:
            codes, _ = model.encode_image(loc.to(device))
            return codes.cpu()
        x = loc[:, [1, 0]].double().to(device) if args.encoder == "satclip" else loc.float().to(device)
        return F.normalize(model(x).float(), dim=-1).cpu()


def encode_split(args, model, split, device, name, entry, ds_kw):
    loader = DataLoader(build_dataset(name, split, entry, **ds_kw),
                        batch_size=args.batch_size, shuffle=False)
    codes, vals, locs = zip(*[
        (_encode(args, model, loc, device), val, loc)
        for loc, val in tqdm(loader, desc=f"Encoding {split}")
    ])
    return torch.cat(codes), torch.cat(vals), torch.cat(locs)


def main():
    args = get_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    paths = load_paths()
    ds_name, *rest = args.dataset.split(".", 1)
    entry = dict(paths["datasets"][ds_name])
    if rest:
        entry = dict(entry[rest[0]])
    ds_kw = dict(
        value_col=entry.pop("value_col"),
        lat_col=entry.pop("lat_col"),
        lon_col=entry.pop("lon_col"),
        **dataset_kwargs(args.dataset),
    )

    model = _load_model(args, paths, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.sparse_model if args.sparse_model else args.encoder

    for split in args.splits:
        codes, vals, locs = encode_split(args, model, split, device, ds_name, entry, ds_kw)
        torch.save({"codes": codes, "vals": vals, "locs": locs}, out_dir / f"{prefix}_{split}.pt")


if __name__ == "__main__":
    main()

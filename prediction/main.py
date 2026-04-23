import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))

from .dataset import GeoDataset, REGRESSION_TASKS, CLASSIFICATION_TASKS
from .sparse_codes import get_sparse_codes
from .model import fit_ridge
from .eval import eval_ridge
from .top_concepts import top_concepts
from .concept_manipulation import run_manipulation


def run_task(task, location_model, splice_model, k_values, device, rng, data_root):
    print(f"\n=== {task} ===")
    train_ds = GeoDataset(task, root=data_root, split="train")
    test_ds  = GeoDataset(task, root=data_root, split="test")

    codes_train = get_sparse_codes(train_ds.latlons, location_model, splice_model, device)
    codes_test  = get_sparse_codes(test_ds.latlons,  location_model, splice_model, device)
    y_train, y_test = train_ds.values, test_ds.values

    probe    = fit_ridge(task, codes_train, y_train)
    baseline = eval_ridge(task, probe, codes_test, y_test)
    print(f"  baseline: {baseline:.4f}")

    records = []
    for k in k_values:
        idx = top_concepts(task, codes_train, y_train, k)
        # For classification, flatten per-class indices into a unique set
        top_idx = np.unique(np.concatenate(list(idx.values()))) if isinstance(idx, dict) else idx
        rand_idx = rng.choice(codes_train.shape[1], len(top_idx), replace=False)

        for cond, val in run_manipulation(task, probe, codes_test, y_test, codes_train, top_idx, rand_idx).items():
            records.append({"task": task, "k": k, "condition": cond,
                            "metric": val, "delta": val - baseline})
    return records


def load_location_model(model_path, device):
    # TODO: mirror the full build_model call from splice.py if projection layers are needed
    ckpt = torch.load(model_path, map_location=device)
    from models.model import build_model
    model_args = argparse.Namespace(**ckpt["args"])
    model_args.precomputed_location_embeddings = False
    model = build_model(
        text_encoder=model_args.text_encoder,
        location_encoder=model_args.location_encoder,
        text_projection=model_args.text_projection,
        location_projection=model_args.location_projection,
        shared_dim=model_args.shared_dim,
        text_finetune_mode=model_args.text_finetune_mode,
        loc_finetune_mode=model_args.loc_finetune_mode,
        text_proj_hidden_layers=model_args.text_proj_hidden_layers,
        text_proj_hidden_features=model_args.text_proj_hidden_features,
        loc_proj_hidden_layers=model_args.loc_proj_hidden_layers,
        loc_proj_hidden_features=model_args.loc_proj_hidden_features,
        text_nonlinearity=model_args.text_nonlinearity,
        loc_nonlinearity=model_args.loc_nonlinearity,
        precomputed_text_embeddings=model_args.precomputed_text_embeddings,
        precomputed_location_embeddings=model_args.precomputed_location_embeddings,
        device=device,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model.location_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splice_model", required=True, help="Path to splice_results/splice_model.pt")
    parser.add_argument("--model_path",   required=True, help="Checkpoint for location-text model")
    parser.add_argument("--data_root",    default="~/data/prediction_tasks")
    parser.add_argument("--k",            nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--tasks",        nargs="+", default=REGRESSION_TASKS + CLASSIFICATION_TASKS)
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    splice_model    = torch.load(args.splice_model, map_location=device)
    location_model  = load_location_model(args.model_path, device)
    rng = np.random.default_rng(42)

    records = []
    for task in args.tasks:
        records.extend(run_task(task, location_model, splice_model, args.k, device, rng, args.data_root))

    df = pd.DataFrame(records)
    df.to_csv("manipulation_results.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
Save an initial (untrained) checkpoint for the GeoClip text encoder configuration.
The output is loadable by load_checkpoint() in main.py via --resume_from.
"""
import argparse
from pathlib import Path

import torch

from loss import make_loss
from main import build_optimizer, save_checkpoint
from models.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="geoclip_init.pt", help="Output checkpoint path")
    # Encoder / projection knobs — match the config you intend to train
    parser.add_argument("--location_encoder", type=str, default="geoclip")
    parser.add_argument("--text_projection", type=str, default="none")
    parser.add_argument("--location_projection", type=str, default="none")
    parser.add_argument("--shared_dim", type=int, default=512)
    parser.add_argument("--text_finetune_mode", type=str, default="only_proj")
    parser.add_argument("--loc_finetune_mode", type=str, default="only_proj")
    parser.add_argument("--text_proj_hidden_layers", type=int, default=1)
    parser.add_argument("--text_proj_hidden_features", type=int, default=512)
    parser.add_argument("--loc_proj_hidden_layers", type=int, default=1)
    parser.add_argument("--loc_proj_hidden_features", type=int, default=512)
    parser.add_argument("--text_nonlinearity", type=str, default=None)
    parser.add_argument("--loc_nonlinearity", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    args = parser.parse_args()

    # geoclip text encoder is always non-precomputed (its forward pass is part of the model)
    args.text_encoder = "geoclip"
    args.precomputed_text_embeddings = False
    args.precomputed_location_embeddings = True

    device = torch.device("cpu")

    model = build_model(
        text_encoder=args.text_encoder,
        location_encoder=args.location_encoder,
        text_projection=args.text_projection,
        location_projection=args.location_projection,
        shared_dim=args.shared_dim,
        text_finetune_mode=args.text_finetune_mode,
        loc_finetune_mode=args.loc_finetune_mode,
        text_proj_hidden_layers=args.text_proj_hidden_layers,
        text_proj_hidden_features=args.text_proj_hidden_features,
        loc_proj_hidden_layers=args.loc_proj_hidden_layers,
        loc_proj_hidden_features=args.loc_proj_hidden_features,
        text_nonlinearity=args.text_nonlinearity,
        loc_nonlinearity=args.loc_nonlinearity,
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
        device=device,
    )

    criterion = make_loss("clip_asymmetric", logit_scale_temp=0.07, lambda_alignment=None, sigma=None).to(device)
    optimizer = build_optimizer(args, model, criterion)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path=out,
        epoch=0,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        best_val_loss=float("inf"),
        args=args,
    )
    print(f"Saved initial GeoClip checkpoint to {out}")


if __name__ == "__main__":
    main()

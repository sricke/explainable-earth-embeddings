import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

import torch
import wandb
from checkpoint import build_checkpoint_path, load_checkpoint
from dataset import build_dataloaders
from loss import CLIPLoss
from models.location_encoder import LOCATION_EMBEDDING_DIMENSIONS
from models.model import build_model
from train import run_train
from utils import EarlyStopping, build_optimizer, build_scheduler, set_seed

_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "location_text_alignment.yaml"


def _load_config() -> dict:
    try:
        return yaml.safe_load(_CONFIG_PATH.read_text()) or {}
    except FileNotFoundError:
        return {}


def get_args():
    cfg = _load_config()
    dataset = (cfg.get("dataset") or {}).get
    model = (cfg.get("model") or {}).get
    text_encoder = (cfg.get("text_encoder") or {}).get
    location_encoder = (cfg.get("location_encoder") or {}).get
    training = (cfg.get("training") or {}).get
    log = (cfg.get("logging") or {}).get

    p = argparse.ArgumentParser()

    # Location encoder is the only required CLI arg
    p.add_argument("--location_encoder", required=True)

    # Dataset
    p.add_argument("--dataset_name", default=dataset("name"))
    p.add_argument("--dataset_path", default=dataset("path"))

    # Text encoder
    p.add_argument("--text_encoder", default=text_encoder("name", "open_clip_vit_l"))
    p.add_argument(
        "--text_finetune_mode", default=text_encoder("finetune_mode", "lora")
    )
    p.add_argument("--text_projection", default=text_encoder("projection", "linear"))
    p.add_argument(
        "--text_proj_hidden_layers",
        type=int,
        default=text_encoder("proj_hidden_layers", 0),
    )
    p.add_argument(
        "--text_proj_hidden_features",
        type=int,
        default=text_encoder("proj_hidden_features"),
    )
    p.add_argument("--text_nonlinearity", default=text_encoder("proj_nonlinearity"))
    p.add_argument(
        "--precomputed_text_embeddings",
        action="store_true",
        default=text_encoder("precomputed", False),
    )
    p.add_argument("--lora_r", type=int, default=text_encoder("lora_r", 8))
    p.add_argument("--lora_alpha", type=float, default=text_encoder("lora_alpha", 1.0))
    p.add_argument(
        "--lora_last_n_layers", type=int, default=text_encoder("lora_last_n_layers", 8)
    )

    # Location encoder
    p.add_argument(
        "--loc_finetune_mode", default=location_encoder("finetune_mode", "only_proj")
    )
    p.add_argument(
        "--precomputed_location_embeddings",
        action="store_true",
        default=location_encoder("precomputed", True),
    )

    # Training
    p.add_argument("--batch_size", type=int, default=training("batch_size", 256))
    p.add_argument("--num_workers", type=int, default=training("num_workers", 4))
    p.add_argument(
        "--train_subsample_size", type=int, default=training("train_subsample_size")
    )
    p.add_argument("--lr", type=float, default=training("lr", 1e-4))
    p.add_argument("--weight_decay", type=float, default=training("weight_decay", 0.01))
    p.add_argument("--max_steps", type=int, default=training("max_steps", 10_000))
    p.add_argument(
        "--accumulation_steps", type=int, default=training("accumulation_steps", 1)
    )
    p.add_argument(
        "--val_every_n_steps", type=int, default=training("val_every_n_steps", 500)
    )
    p.add_argument("--scheduler", default=training("scheduler"))
    p.add_argument("--warmup_steps", type=int, default=training("warmup_steps", 0))
    p.add_argument(
        "--num_val_checks_without_improvement",
        type=int,
        default=training("num_val_checks_without_improvement", -1),
    )
    p.add_argument("--seed", type=int, default=training("seed", 42))

    # Model / logging
    p.add_argument("--model_save_dir", default=model("save_dir"))
    p.add_argument("--resume", default=log("resume"))
    p.add_argument("--wandb_project", default=log("wandb_project"))

    args = p.parse_args()

    if args.dataset_path is None:
        p.error("dataset.path must be set in config or via --dataset_path")
    if args.model_save_dir is None:
        p.error("model.save_dir must be set in config or via --model_save_dir")

    return args


def main():
    args = get_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = Path(args.dataset_path).expanduser()
    shared_dim = LOCATION_EMBEDDING_DIMENSIONS[args.location_encoder]

    if args.wandb_project:
        wandb.init(project=args.wandb_project, config=vars(args))

    train_dataloader, val_dataloader = build_dataloaders(
        dataset_path,
        args.batch_size,
        args.num_workers,
        args.seed,
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
        train_subsample_size=args.train_subsample_size,
    )

    model = build_model(
        args.text_encoder,
        args.location_encoder,
        args.text_projection,
        shared_dim,
        args.text_finetune_mode,
        args.loc_finetune_mode,
        args.text_proj_hidden_layers,
        args.text_proj_hidden_features,
        args.text_nonlinearity,
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_last_n_layers=args.lora_last_n_layers,
    )
    criterion = CLIPLoss().to(device)
    optimizer = build_optimizer(args, model, criterion)
    scheduler = build_scheduler(args, optimizer, args.max_steps)
    early_stopper = (
        EarlyStopping(patience_steps=args.num_val_checks_without_improvement)
        if args.num_val_checks_without_improvement > 0
        else None
    )

    checkpoint_path = build_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint path: {checkpoint_path}")

    start_step, best_val = 0, float("inf")
    if args.resume:
        start_step, best_val = load_checkpoint(
            args.resume, model, criterion, optimizer, scheduler, device
        )
        if early_stopper:
            early_stopper.load_best(best_val)

    run_train(
        train_dataloader,
        val_dataloader,
        model,
        criterion,
        optimizer,
        device,
        save_path=checkpoint_path,
        args=args,
        scheduler=scheduler,
        accumulation_steps=args.accumulation_steps,
        max_steps=args.max_steps,
        val_every_n_steps=args.val_every_n_steps,
        early_stopper=early_stopper,
        start_step=start_step,
        best_val=best_val,
    )

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()

import argparse
import logging
from pathlib import Path

import torch
import wandb
import yaml

from dataset import build_dataloaders
from models.model import build_model
from models.finetune import apply_lora
from loss import make_loss
from train import train_epoch
from eval import val_epoch
from utils import EarlyStopping, set_seed
from log_utils import setup_logging, build_run_name


logger = logging.getLogger(__name__)


def get_args():
    # Check for --config file before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        return argparse.Namespace(**cfg)

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--train_subsample_size', type=int, default=None)
    parser.add_argument('--val_subsample_size', type=int, default=None)
    parser.add_argument('--precomputed_text_embeddings', type=bool, default=True)
    parser.add_argument('--precomputed_location_embeddings', type=bool, default=True)
    # Encoders
    parser.add_argument('--text_encoder', type=str, required=True, help="'open_clip' | 'geoclip'")
    parser.add_argument('--location_encoder', type=str, required=True, help="'satclip' | 'geoclip'")
    parser.add_argument('--text_finetune_mode', type=str, default='only_proj', help="'all' | 'lora' | 'only_proj'")
    parser.add_argument('--loc_finetune_mode', type=str, default='only_proj', help="'all' | 'lora' | 'only_proj'")
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank (used when finetune_mode=lora)')
    # Projections
    parser.add_argument('--text_projection', type=str, default='linear', help="'linear' | 'mlp'")
    parser.add_argument('--text_proj_hidden_layers', type=int, default=1)
    parser.add_argument('--text_proj_hidden_features', type=int, default=512)
    parser.add_argument('--location_projection', type=str, default='linear', help="'none' | 'linear' | 'mlp'")
    parser.add_argument('--loc_proj_hidden_layers', type=int, default=1)
    parser.add_argument('--loc_proj_hidden_features', type=int, default=512)
    parser.add_argument('--shared_dim', type=int, default=256)
    # Loss
    parser.add_argument('--train_loss', type=str, required=True, help="'clip' | 'mse'")
    parser.add_argument('--lambda_alignment', type=float)
    parser.add_argument('--sigma', type=float)
    parser.add_argument('--logit_scale_temp', type=float, default=0.07)
    # Training
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--text_nonlinearity', type=str, default=None, help="'relu' | 'sine'")
    parser.add_argument('--loc_nonlinearity', type=str, default=None, help="'relu' | 'sine'")
    parser.add_argument('--accumulation_steps', type=int, default=1, help="1 disables accumulation")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    # Checkpointing
    parser.add_argument(
        '--model_save_path', type=str, default=None,
        help='Base directory for checkpoints (default: ~/outputs/explainable-earth-embeddings/<dataset_name>); '
             'checkpoints go in <base>/<run-subdir>/best.pt. File logs use ./logs in the cwd.',
    )
    parser.add_argument('--resume_from', type=str)
    # Logging
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B team or username (optional; defaults to logged-in default).')
    parser.add_argument('--wandb_project', type=str, default='explainable-earth-embeddings')
    parser.add_argument('--wandb_run_name', type=str)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()



def build_optimizer(args, model, logit_scale):
    return torch.optim.AdamW(
        list(model.parameters()) + [logit_scale],
        lr=args.lr, weight_decay=args.weight_decay,
    )


def save_checkpoint(path, epoch, model, optimizer, logit_scale, best_val_loss, args):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'logit_scale': logit_scale.data,
        'best_val_loss': best_val_loss,
        'args': vars(args),
    }, path)


def load_checkpoint(path, model, optimizer, logit_scale, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    logit_scale.data = ckpt['logit_scale'].to(device)
    model.to(device)
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)
    logger.info(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf'))


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset_name = Path(args.dataset_path).expanduser().resolve().parent.parent.name
    default_wandb_name, default_run_tag = build_run_name(args, dataset_name)
    desired_wandb_name = str(args.wandb_run_name).strip().replace(" ", "-").replace("/", "_").replace("\\", "_") if getattr(args, "wandb_run_name", None) else default_wandb_name

    # skip run if all epochs are completed
    _early_checkpoint_base = (Path(args.model_save_path).expanduser().resolve() if getattr(args, "model_save_path", None) else (Path.home() / "outputs" / "explainable-earth-embeddings" / dataset_name).resolve())
    if (_early_checkpoint_base / default_run_tag / "done").exists():
        logger.info(f"Run {default_run_tag} already completed. Skipping.")
        return

    if not args.no_wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=desired_wandb_name,
            config=vars(args),
        )
        wandb_name = wandb.run.name if wandb.run is not None else desired_wandb_name
        run_tag = wandb_name.replace(f"{dataset_name}__", "", 1).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")
    else:
        run_tag = default_run_tag

    run_tag = str(run_tag).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")
    setup_logging(f"{dataset_name}__{run_tag}".strip().replace(" ", "-").replace("/", "_").replace("\\", "_"), Path("logs"))

    if getattr(args, "model_save_path", None):
        checkpoint_base = Path(args.model_save_path).expanduser().resolve()
    else:
        checkpoint_base = (Path.home() / "outputs" / "explainable-earth-embeddings" / dataset_name).resolve()
    checkpoint_dir = checkpoint_base / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best.pt"
    config_save_path = checkpoint_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Checkpoints directory: {checkpoint_dir}")

    args.precomputed_text_embeddings = args.precomputed_text_embeddings and args.text_finetune_mode not in ("all", "lora")
    args.precomputed_location_embeddings = args.precomputed_location_embeddings and args.loc_finetune_mode not in ("all", "lora")

    train_dataloader, val_dataloader = build_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        precomputed_text_embeddings=args.precomputed_text_embeddings,
        precomputed_location_embeddings=args.precomputed_location_embeddings,
        train_subsample_size=args.train_subsample_size,
        val_subsample_size=args.val_subsample_size,
    )
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
    if args.text_finetune_mode == 'lora':
        model.text_encoder.text_encoder.m = apply_lora(model.text_encoder.text_encoder.m, args.lora_rank)
    if not args.precomputed_text_embeddings:
        model.text_encoder.text_encoder.m.gradient_checkpointing_enable()

    logit_scale = torch.nn.Parameter(torch.tensor(1.0 / args.logit_scale_temp, device=device).log())
    criterion = make_loss(args.train_loss, logit_scale, args.lambda_alignment, args.sigma)
    optimizer = build_optimizer(args, model, logit_scale)
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    logger.info(f"Loss: {args.train_loss}")
    logger.info(f"LR={args.lr}, weight_decay={args.weight_decay}")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume_from:
        start_epoch, best_val_loss = load_checkpoint(args.resume_from, model, optimizer, logit_scale, device)
        early_stopping.load_best(best_val_loss)
    elif save_path.exists() and not (checkpoint_dir / "done").exists():
        logger.info(f"Found existing checkpoint without 'done' marker. Auto-resuming from {save_path}")
        start_epoch, best_val_loss = load_checkpoint(save_path, model, optimizer, logit_scale, device)
        early_stopping.load_best(best_val_loss)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, logger, accumulation_steps=args.accumulation_steps)
        val_loss = val_epoch(val_dataloader, model, criterion, epoch, device, logger)

        logger.info(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_path, epoch, model, optimizer, logit_scale, best_val_loss, args)
            logger.info(f"Saved checkpoint (val={val_loss:.4f})")

        if early_stopping(val_loss):
            print("Early Stopping")
            (checkpoint_dir / "done").touch()
            logger.info("Training stopped early. Wrote 'done' marker.")
            break
    else:

        (checkpoint_dir / "done").touch()
        logger.info("Training completed all epochs. Wrote 'done' marker.")

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

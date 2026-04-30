import argparse
import logging
from pathlib import Path

import torch
import wandb
import yaml

from dataset import build_dataloaders
from models.model import build_model
from models.queue import LocQueue, TextQueue
from models.text_encoder import TEXT_EMBEDDING_DIMENSIONS
from models.finetune import apply_lora
from loss import make_loss
from train import train_epoch
from eval import val_epoch
from utils import EarlyStopping, set_seed, build_scheduler
from log_utils import setup_logging, build_run_name


logger = logging.getLogger(__name__)


def sanitize(s: str) -> str:
    return str(s).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")


def get_args():
    # Check for --config file before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str)
    pre_args, _ = pre.parse_known_args()

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f)
        return argparse.Namespace(**cfg)

    _int = lambda x: None if x == 'None' else int(x)
    _float = lambda x: None if x == 'None' else float(x)

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--precomputed_dir', type=str, required=True)
    parser.add_argument('--train_subsample_size', type=_int, required=True)
    parser.add_argument('--val_subsample_size', type=_int, required=True)
    parser.add_argument('--precomputed_text_embeddings', type=bool, required=True)
    parser.add_argument('--precomputed_location_embeddings', type=bool, required=True)
    # Encoders
    parser.add_argument('--text_encoder', type=str, required=True, help="'open_clip' | 'geoclip'")
    parser.add_argument('--location_encoder', type=str, required=True, help="'satclip' | 'geoclip'")
    parser.add_argument('--text_finetune_mode', type=str, required=True, help="'all' | 'lora' | 'only_proj'")
    parser.add_argument('--loc_finetune_mode', type=str, required=True, help="'all' | 'lora' | 'only_proj'")
    parser.add_argument('--lora_rank', required=True, help='LoRA rank (used when finetune_mode=lora)')
    parser.add_argument('--lora_layers', required=True, type=int, help='Apply LoRA to last N transformer blocks (None = all)')
    # Projections
    parser.add_argument('--text_projection', type=str, required=True, help="'linear' | 'mlp'")
    parser.add_argument('--text_proj_hidden_layers', type=int, required=True)
    parser.add_argument('--text_proj_hidden_features', type=int, required=True)
    parser.add_argument('--location_projection', type=str, required=True, help="'none' | 'linear' | 'mlp'")
    parser.add_argument('--loc_proj_hidden_layers', type=int, required=True)
    parser.add_argument('--loc_proj_hidden_features', type=int, required=True)
    parser.add_argument('--shared_dim', type=int, required=True)
    # Loss
    parser.add_argument('--train_loss', type=str, required=True, help="'clip' | 'mse'")
    parser.add_argument('--lambda_alignment', type=_float, required=True)
    parser.add_argument('--sigma', type=_float, required=True)
    parser.add_argument('--logit_scale_temp', type=float, required=True)
    # Training
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--patience', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--scheduler', type=str, required=True, help="'cosine' | None")
    parser.add_argument('--warmup_steps', type=int, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--text_nonlinearity', type=str, required=True, help="'relu' | 'sine'")
    parser.add_argument('--loc_nonlinearity', type=str, required=True, help="'relu' | 'sine'")
    parser.add_argument('--accumulation_steps', type=int, required=True, help="1 disables accumulation")
    parser.add_argument('--loc_queue_size', type=_int, required=True, help="Queue size for asymmetric loss (location side); None disables")
    parser.add_argument('--text_queue_size', type=_int, required=True, help="Queue size for asymmetric loss (text side); None disables")
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, required=True)
    # Checkpointing
    parser.add_argument(
        '--model_save_path', type=str, required=True,
        help='Base directory for checkpoints; checkpoints go in <base>/<run-subdir>/best.pt. File logs use ./logs in the cwd.',
    )
    parser.add_argument('--resume_from', type=str, required=True)
    # Logging
    parser.add_argument('--wandb_entity', type=str, required=True,
                        help='W&B team or username.')
    parser.add_argument('--wandb_project', type=str, required=True)
    parser.add_argument('--wandb_run_name', type=str, required=True)
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def build_checkpoint_base(args, dataset_name: str) -> Path:
    if getattr(args, "model_save_path", None):
        return Path(args.model_save_path).expanduser().resolve()
    return (Path(__file__).parent / "../../outputs" / "explainable-earth-embeddings" / dataset_name).resolve()


def build_optimizer(args, model, criterion):
    no_decay = lambda n, p: p.ndim < 2 or any(k in n for k in ("bn", "ln", "bias", "logit_scale"))
    named = list(model.named_parameters()) + list(criterion.named_parameters())
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in named if no_decay(n, p) and p.requires_grad], "weight_decay": 0.0},
            {"params": [p for n, p in named if not no_decay(n, p) and p.requires_grad], "weight_decay": args.weight_decay},
        ],
        lr=args.lr,
    )


def save_checkpoint(path, epoch, model, criterion, optimizer, scheduler, best_val_loss, args, loc_queue=None, text_queue=None):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'criterion': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'args': vars(args),
        'loc_queue': loc_queue.state_dict() if loc_queue is not None else None,
        'text_queue': text_queue.state_dict() if text_queue is not None else None,
    }, path)


def load_checkpoint(path, model, criterion, optimizer, scheduler, device, loc_queue=None, text_queue=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if 'criterion' in ckpt:
        criterion.load_state_dict(ckpt['criterion'])
    elif 'logit_scale' in ckpt and hasattr(criterion, 'logit_scale'):
        # backward compat: old checkpoints saved logit_scale separately
        criterion.logit_scale.data = ckpt['logit_scale'].to(device)
    optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])
    if loc_queue is not None and ckpt.get('loc_queue') is not None:
        loc_queue.load_state_dict(ckpt['loc_queue'])
        loc_queue.to(device)
    if text_queue is not None and ckpt.get('text_queue') is not None:
        text_queue.load_state_dict(ckpt['text_queue'])
        text_queue.to(device)
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
    desired_wandb_name = sanitize(args.wandb_run_name) if getattr(args, "wandb_run_name", None) else default_wandb_name

    checkpoint_base = build_checkpoint_base(args, dataset_name)
    if (checkpoint_base / default_run_tag / "done").exists():
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
        run_tag = sanitize(wandb_name.replace(f"{dataset_name}__", "", 1))
    else:
        run_tag = default_run_tag

    run_tag = sanitize(run_tag)
    setup_logging(sanitize(f"{dataset_name}__{run_tag}"), Path("logs"))

    checkpoint_dir = checkpoint_base / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best.pt"
    with open(checkpoint_dir / "config.yaml", "w") as f:
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
        model.text_encoder.text_encoder.m = apply_lora(
            model.text_encoder.text_encoder.m, int(args.lora_rank), last_n_layers=args.lora_layers
        )
    if not args.precomputed_text_embeddings:
        model.text_encoder.text_encoder.m.gradient_checkpointing_enable()

    loc_queue = None
    if getattr(args, "loc_queue_size", None):
        queue_dim = 2 if not args.precomputed_location_embeddings else model.location_encoder.location_embedding_dim
        loc_queue = LocQueue(queue_size=args.loc_queue_size, dim=queue_dim).to(device)

    text_queue = None
    if getattr(args, "text_queue_size", None):
        text_queue_dim = TEXT_EMBEDDING_DIMENSIONS[args.text_encoder] if args.precomputed_text_embeddings else model.text_encoder.output_dim
        text_queue = TextQueue(queue_size=args.text_queue_size, dim=text_queue_dim).to(device)

    criterion = make_loss(args.train_loss, args.logit_scale_temp, args.lambda_alignment, args.sigma).to(device)
    optimizer = build_optimizer(args, model, criterion)
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = build_scheduler(args, optimizer, total_steps)
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    logger.info(f"Loss: {args.train_loss}")
    logger.info(f"LR={args.lr}, scheduler={getattr(args, 'scheduler', None)}, warmup_steps={getattr(args, 'warmup_steps', 0)}, total_steps={total_steps}, weight_decay={args.weight_decay}")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume_from:
        start_epoch, best_val_loss = load_checkpoint(args.resume_from, model, criterion, optimizer, scheduler, device, loc_queue=loc_queue, text_queue=text_queue)
        early_stopping.load_best(best_val_loss)
    elif save_path.exists() and not (checkpoint_dir / "done").exists():
        logger.info(f"Found existing checkpoint without 'done' marker. Auto-resuming from {save_path}")
        start_epoch, best_val_loss = load_checkpoint(save_path, model, criterion, optimizer, scheduler, device, loc_queue=loc_queue, text_queue=text_queue)
        early_stopping.load_best(best_val_loss)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, logger, scheduler=scheduler, accumulation_steps=args.accumulation_steps, loc_queue=loc_queue, text_queue=text_queue)
        val_loss = val_epoch(val_dataloader, model, criterion, epoch, device, logger, loc_queue=loc_queue, text_queue=text_queue)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")
        if wandb.run is not None:
            wandb.log({"train/lr": current_lr, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_path, epoch, model, criterion, optimizer, scheduler, best_val_loss, args, loc_queue=loc_queue, text_queue=text_queue)
            logger.info(f"Saved checkpoint (val={val_loss:.4f})")

        if early_stopping(val_loss):
            logger.info("Early stopping triggered.")
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

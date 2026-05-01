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
from utils import EarlyStopping, set_seed, build_scheduler, build_optimizer
from log_utils import setup_logging, build_run_name, sanitize
from checkpoint import build_checkpoint_base, save_checkpoint, load_checkpoint


logger = logging.getLogger(__name__)


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
    parser.add_argument('--val_every_n_steps', type=int, required=True)
    parser.add_argument('--num_val_checks_without_improvement', type=int, required=True)
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


def init_run(args, dataset_name: str, default_run_tag: str, desired_wandb_name: str) -> str:
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
    return sanitize(run_tag)


def setup_checkpoint_dir(checkpoint_base: Path, run_tag: str, args) -> tuple[Path, Path]:
    checkpoint_dir = checkpoint_base / run_tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / "best.pt"
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    logger.info(f"Checkpoints directory: {checkpoint_dir}")
    return checkpoint_dir, save_path


def build_queues(args, model, device):
    loc_queue = None
    if getattr(args, "loc_queue_size", None):
        queue_dim = 2 if not args.precomputed_location_embeddings else model.location_encoder.location_embedding_dim
        loc_queue = LocQueue(queue_size=args.loc_queue_size, dim=queue_dim).to(device)

    text_queue = None
    if getattr(args, "text_queue_size", None):
        text_queue_dim = TEXT_EMBEDDING_DIMENSIONS[args.text_encoder] if args.precomputed_text_embeddings else model.text_encoder.output_dim
        text_queue = TextQueue(queue_size=args.text_queue_size, dim=text_queue_dim).to(device)

    return loc_queue, text_queue


def try_resume(args, save_path: Path, checkpoint_dir: Path, model, criterion, optimizer, scheduler, device, loc_queue, text_queue, early_stopping):
    start_epoch = 0
    best_val_loss = float('inf')
    global_optimizer_step = 0

    if args.resume_from:
        start_epoch, best_val_loss, global_optimizer_step = load_checkpoint(args.resume_from, model, criterion, optimizer, scheduler, device, loc_queue=loc_queue, text_queue=text_queue)
        early_stopping.load_best(best_val_loss)
    elif save_path.exists() and not (checkpoint_dir / "done").exists():
        logger.info(f"Found existing checkpoint without 'done' marker. Auto-resuming from {save_path}")
        start_epoch, best_val_loss, global_optimizer_step = load_checkpoint(save_path, model, criterion, optimizer, scheduler, device, loc_queue=loc_queue, text_queue=text_queue)
        early_stopping.load_best(best_val_loss)

    return start_epoch, best_val_loss, global_optimizer_step


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

    run_tag = init_run(args, dataset_name, default_run_tag, desired_wandb_name)
    setup_logging(sanitize(f"{dataset_name}__{run_tag}"), Path("logs"))
    checkpoint_dir, save_path = setup_checkpoint_dir(checkpoint_base, run_tag, args)

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

    loc_queue, text_queue = build_queues(args, model, device)

    criterion = make_loss(args.train_loss, args.logit_scale_temp, args.lambda_alignment, args.sigma).to(device)
    optimizer = build_optimizer(args, model, criterion)
    steps_per_epoch = len(train_dataloader) // args.accumulation_steps
    total_steps = args.num_epochs * steps_per_epoch
    scheduler = build_scheduler(args, optimizer, total_steps)
    early_stopping = EarlyStopping(patience=args.num_val_checks_without_improvement, mode="min")
    logger.info(f"Loss: {args.train_loss}")
    logger.info(f"LR={args.lr}, scheduler={getattr(args, 'scheduler', None)}, warmup_steps={getattr(args, 'warmup_steps', 0)}, total_steps={total_steps}, weight_decay={args.weight_decay}")

    start_epoch, best_val_loss, global_optimizer_step = try_resume(
        args, save_path, checkpoint_dir, model, criterion, optimizer, scheduler, device, loc_queue, text_queue, early_stopping
    )

    def val_callback(global_opt_step: int) -> bool:
        nonlocal best_val_loss
        val_loss = val_epoch(val_dataloader, model, criterion, epoch, device, logger, loc_queue=loc_queue, text_queue=text_queue)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Step {global_opt_step} | val={val_loss:.4f} | lr={current_lr:.2e}")
        if wandb.run is not None:
            wandb.log({"val/loss": val_loss, "train/lr": current_lr, "step": global_opt_step})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_path, epoch, model, criterion, optimizer, scheduler, best_val_loss, args, global_optimizer_step=global_opt_step, loc_queue=loc_queue, text_queue=text_queue)
            logger.info(f"Saved checkpoint (val={val_loss:.4f})")
        return early_stopping(val_loss)

    for epoch in range(start_epoch, args.num_epochs):
        train_loss, steps_taken, should_stop = train_epoch(
            train_dataloader, model, criterion, optimizer, epoch, device, logger,
            scheduler=scheduler, accumulation_steps=args.accumulation_steps,
            loc_queue=loc_queue, text_queue=text_queue,
            val_callback=val_callback, val_every_n_steps=args.val_every_n_steps,
            global_step_offset=global_optimizer_step,
        )
        global_optimizer_step += steps_taken

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} | train={train_loss:.4f} | lr={current_lr:.2e}")
        if wandb.run is not None:
            wandb.log({"train/loss_epoch": train_loss, "train/lr": current_lr, "epoch": epoch})

        if should_stop:
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

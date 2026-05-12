import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def build_checkpoint_path(args) -> Path:
    """
    Checkpoint path
    """
    tft = args.text_finetune_mode
    if tft == "lora":
        tft = f"lora-r{args.lora_r}"

    architecture_parts = "_".join([
        f"txt-{args.text_encoder}",
        f"tproj-{args.text_projection}",
        f"loc-{args.location_encoder}",
    ])

    train_parts = [
        f"tft-{tft}",
        f"lft-{args.loc_finetune_mode}",
        f"lr-{args.lr}",
        f"seed-{args.seed}",
    ]
    if args.train_subsample_size is not None:
        train_parts.append(f"sub-{args.train_subsample_size}")
    train = "_".join(train_parts)

    base = Path(args.model_save_dir).expanduser().resolve()
    return base / args.dataset_name / architecture_parts / train / "best.pt"

def save_checkpoint(path, step, model, criterion, optimizer, scheduler, best_val_loss, args):
    """
    Save checkpoint
    """
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'criterion': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }, path)


def load_checkpoint(path, model, criterion, optimizer, scheduler, device):
    """
    Load checkpoint from path
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])

    if 'criterion' in ckpt:
        criterion.load_state_dict(ckpt['criterion'])
    elif 'logit_scale' in ckpt and hasattr(criterion, 'logit_scale'):
        criterion.logit_scale.data = ckpt['logit_scale'].to(device)

    optimizer.load_state_dict(ckpt['optimizer'])

    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])

    model.to(device)
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)

    step = ckpt.get('step')
    logger.info(f"Resumed from step {step}")
    return step, ckpt.get('best_val_loss', float('inf'))

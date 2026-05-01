import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def build_checkpoint_base(args, dataset_name: str) -> Path:
    if getattr(args, "model_save_path", None):
        return Path(args.model_save_path).expanduser().resolve()
    return (Path(__file__).parent / "../../outputs" / "explainable-earth-embeddings" / dataset_name).resolve()


def save_checkpoint(path, epoch, model, criterion, optimizer, scheduler, best_val_loss, args, global_optimizer_step=0, loc_queue=None, text_queue=None):
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'criterion': criterion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'global_optimizer_step': global_optimizer_step,
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
        # old checkpoints saved logit_scale separately
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
    return ckpt['epoch'] + 1, ckpt.get('best_val_loss', float('inf')), ckpt.get('global_optimizer_step', 0)

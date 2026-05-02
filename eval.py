import torch
import wandb

from queue_utils import _get_loc_queue_embeddings, _get_text_queue_embeddings

@torch.no_grad()
def run_eval(val_dataloader, model, criterion, global_step, device, epoch=0, loc_queue=None, text_queue=None):
    model.eval()
    total_loss = 0.0

    for i, batch in enumerate(val_dataloader):
        batch_loss = val_step(batch, model, criterion, device, loc_queue=loc_queue, text_queue=text_queue)

        if wandb.run is not None:
            wandb.log({"val/loss_step": batch_loss, "step": global_step})

        total_loss += batch_loss

    epoch_loss = total_loss / len(val_dataloader)
    if wandb.run is not None:
        wandb.log({"val/loss_epoch": epoch_loss, "epoch": epoch})

    return epoch_loss

@torch.no_grad()
def val_step(batch, model, criterion, device, loc_queue=None, text_queue=None) -> float:
    locs, texts = batch

    if not isinstance(locs, torch.Tensor):
        raise TypeError(f"Expected locs to be a torch.Tensor, got {type(locs)}")
    if not isinstance(texts, (torch.Tensor, list, tuple)):
        raise TypeError(f"Expected texts to be a torch.Tensor, list, or tuple, got {type(texts)}")
    locs = locs.to(device)
    if isinstance(texts, torch.Tensor):
        texts = texts.to(device)

    text_features, location_features = model(texts, locs)
    loc_queue_embeddings = _get_loc_queue_embeddings(loc_queue, model, device) if loc_queue is not None else None
    text_queue_embeddings = _get_text_queue_embeddings(text_queue, model, device) if text_queue is not None else None
    step_loss = criterion(text_features, location_features, loc_queue=loc_queue_embeddings, text_queue=text_queue_embeddings)

    return step_loss.item()

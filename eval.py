import torch
import wandb
from tqdm import tqdm

from train import _get_loc_queue_embeddings, _get_text_queue_embeddings


@torch.no_grad()
def val_epoch(val_dataloader, model, criterion, epoch, device, logger, loc_queue=None, text_queue=None) -> float:
    model.eval()
    total_loss = 0.0
    global_step = epoch * len(val_dataloader)

    bar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))

    for i, (locs, texts) in bar:
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
        loss = criterion(text_features, location_features, loc_queue=loc_queue_embeddings, text_queue=text_queue_embeddings)

        loss_value = loss.item()
        total_loss += loss_value
        bar.set_description("Epoch {} Val loss: {:.5f}".format(epoch, loss_value))
        if wandb.run is not None:
            wandb.log({"val/loss_step": loss_value, "val_step": global_step + i})

    epoch_loss = total_loss / len(val_dataloader)
    logger.info(f"[Epoch {epoch}] val_loss={epoch_loss:.4f}")
    if wandb.run is not None:
        wandb.log({"val/loss_epoch": epoch_loss, "epoch": epoch})
    return epoch_loss

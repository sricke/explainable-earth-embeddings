import torch
import wandb
from tqdm import tqdm


@torch.no_grad()
def val_epoch(val_dataloader, model, criterion, epoch, device, logger) -> float:
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
        loss = criterion(text_features, location_features)

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

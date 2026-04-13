import torch
import wandb
from tqdm import tqdm


@torch.no_grad()
def val_epoch(val_dataloader, model, criterion, epoch, device) -> float:
    model.eval()
    total_loss = 0.0
    global_step = epoch * len(val_dataloader)

    for i, (locs, texts) in enumerate(tqdm(val_dataloader, desc=f"Val epoch {epoch}")):
        locs = locs.to(device)
        text_features, location_features = model(texts, locs)
        loss = criterion(text_features, location_features).item()
        total_loss += loss
        wandb.log({"val/loss_step": loss, "val_step": global_step + i})

    epoch_loss = total_loss / len(val_dataloader)
    wandb.log({"val/loss_epoch": epoch_loss, "epoch": epoch})
    return epoch_loss

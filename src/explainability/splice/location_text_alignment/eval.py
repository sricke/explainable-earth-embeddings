import torch
import wandb


@torch.no_grad()
def run_eval(val_dataloader, model, criterion, global_step, device, epoch=0):
    """
    Run eval for all batches in val_loader
    """
    model.eval()
    total_loss = 0.0

    for _, batch in enumerate(val_dataloader):
        batch_loss = val_step(batch, model, criterion, device)

        if wandb.run is not None:
            wandb.log({"val/loss_step": batch_loss, "step": global_step})

        total_loss += batch_loss

    epoch_loss = total_loss / len(val_dataloader)
    if wandb.run is not None:
        wandb.log({"val/loss_epoch": epoch_loss, "epoch": epoch})

    return epoch_loss


@torch.no_grad()
def val_step(batch, model, criterion, device) -> float:
    """
    Run eval for single step of val loader
    """
    locs, texts = batch

    if not isinstance(locs, torch.Tensor):
        raise TypeError(f"Expected locs to be a torch.Tensor, got {type(locs)}")
    if not isinstance(texts, (torch.Tensor, list, tuple)):
        raise TypeError(
            f"Expected texts to be a torch.Tensor, list, or tuple, got {type(texts)}"
        )
    locs = locs.to(device)
    if isinstance(texts, torch.Tensor):
        texts = texts.to(device)

    text_features, location_features = model(texts, locs)

    step_loss = criterion(text_features, location_features)

    return step_loss.item()

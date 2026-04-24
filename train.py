import wandb
import torch
from tqdm import tqdm
from typing import Tuple

def train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, logger, scheduler=None, accumulation_steps=1) -> float:
    logger.info(f"Starting epoch {epoch}")

    model.train()
    total_loss = 0.0
    global_step = epoch * len(train_dataloader)
    optimizer.zero_grad()

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (locs, texts) in bar:
        if not isinstance(locs, torch.Tensor):
            raise TypeError(f"Expected locs to be a torch.Tensor, got {type(locs)}")
        if not isinstance(texts, (torch.Tensor, list, tuple)):
            raise TypeError(f"Expected texts to be a torch.Tensor, list, or tuple, got {type(texts)}")
        locs = locs.to(device)
        if isinstance(texts, torch.Tensor):
            texts = texts.to(device)
        text_features, location_features = model(texts, locs)

        loss = criterion(text_features, location_features) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()

        loss_value = loss.item() * accumulation_steps
        total_loss += loss_value
        bar.set_description("Epoch {} Train loss: {:.5f}".format(epoch, loss_value))
        if wandb.run is not None:
            wandb.log({"train/loss_step": loss_value, "step": global_step + i})

    if scheduler is not None:
        scheduler.step()

    epoch_loss = total_loss / len(train_dataloader)
    logger.info(f"[Epoch {epoch}] train_loss={epoch_loss:.4f}")
    if wandb.run is not None:
        wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch})
    return epoch_loss
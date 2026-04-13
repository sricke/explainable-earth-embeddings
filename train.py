import wandb
import torch
from tqdm import tqdm

def train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, scheduler=None) -> float:
    print("Starting Epoch", epoch)

    model.train()
    total_loss = 0.0
    global_step = epoch * len(train_dataloader)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    for i, (locs, texts) in bar:
        locs = locs.to(device)
        text_features, location_features = model(texts, locs)

        loss = criterion(text_features, location_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))
        wandb.log({"train/loss_step": loss.item(), "step": global_step + i})

    if scheduler is not None:
        scheduler.step()

    epoch_loss = total_loss / len(train_dataloader)
    wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch})
    return epoch_loss
import wandb
import torch
from tqdm import tqdm


@torch.no_grad()
def _get_loc_queue_embeddings(loc_queue, model, device) -> torch.Tensor:
    queue = loc_queue.get().to(device)
    return model.location_model_predict(queue)


@torch.no_grad()
def _get_text_queue_embeddings(text_queue, model, device) -> torch.Tensor:
    queue = text_queue.get().to(device)
    return model.text_model_predict(queue)


def train_epoch(train_dataloader, model, criterion, optimizer, epoch, device, logger, scheduler=None, accumulation_steps=1, loc_queue=None, text_queue=None) -> float:
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

        loc_queue_embeddings = _get_loc_queue_embeddings(loc_queue, model, device) if loc_queue is not None else None
        text_queue_embeddings = _get_text_queue_embeddings(text_queue, model, device) if text_queue is not None else None
        loss = criterion(text_features, location_features, loc_queue=loc_queue_embeddings, text_queue=text_queue_embeddings) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        if loc_queue is not None:
            loc_queue.enqueue(locs.detach())
        if text_queue is not None:
            if isinstance(texts, torch.Tensor):
                text_queue.enqueue(texts.detach())
            else:
                with torch.no_grad():
                    pre_proj = model.text_encoder.encode_texts(texts)
                text_queue.enqueue(pre_proj.detach())

        loss_value = loss.item() * accumulation_steps
        total_loss += loss_value
        bar.set_description("Epoch {} Train loss: {:.5f}".format(epoch, loss_value))
        if wandb.run is not None:
            wandb.log({"train/loss_step": loss_value, "step": global_step + i})

    epoch_loss = total_loss / len(train_dataloader)
    logger.info(f"[Epoch {epoch}] train_loss={epoch_loss:.4f}")
    if wandb.run is not None:
        wandb.log({"train/loss_epoch": epoch_loss, "epoch": epoch})
    return epoch_loss

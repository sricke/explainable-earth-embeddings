import torch
import wandb
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

from eval import run_eval
from checkpoint import save_checkpoint

def run_train(
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    device,
    save_path,
    args,
    scheduler=None,
    accumulation_steps=1,
    max_steps=None,
    val_every_n_steps=100,
    early_stopper=None,
    start_step=0,
    best_val=float("inf"),
):
    """
    Entire training run. Metrics are computed w.r.t steps (as standard for CLIP training)
    """
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(
        total=max_steps,
        initial=start_step,
        desc="training",
    )

    # Track global step
    global_step = start_step
    epoch = 0
    data_iter = iter(train_dataloader)

    while global_step < max_steps:

        # Try to load next batch
        try:
            batch = next(data_iter)
        # If next batch indicates stop, start over for train dataloader
        except StopIteration:
            data_iter = iter(train_dataloader)
            epoch += 1
            continue

        loss, _, _ = train_step(batch, model, criterion, device)

        # Backward for each step
        (loss / accumulation_steps).backward()

        if (global_step + 1) % accumulation_steps == 0:
            optimizer.step() # step the optimizer after accumulation steps
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        global_step += 1
        pbar.update(1)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "epoch": epoch,
        })

        # logging
        if wandb.run is not None:
            wandb.log({
                "train/loss_step": loss.item(),
                "step": global_step,
                "epoch": epoch,
            })

        # Validation
        if global_step % val_every_n_steps == 0:
            val_loss = run_eval(
                val_dataloader,
                model,
                criterion,
                global_step,
                device,
                epoch=epoch
            )

            print(f"Step={global_step}, Val Loss={val_loss}")
            logger.info(f"step={global_step} val={val_loss:.4f}")

            if wandb.run is not None:
                wandb.log({
                    "val/loss": val_loss,
                    "step": global_step,
                })

            model.train()

            # Record best checkpoint
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(save_path, global_step, model, criterion, optimizer, scheduler, best_val, args)

            # Check for early stopping
            if early_stopper is not None and early_stopper(val_loss):
                logger.info(
                    f"Early stopping triggered at step {global_step} "
                    f"(best_val={best_val:.4f})"
                )
                break

    pbar.close()

    return global_step, best_val


def train_step(batch, model, criterion, device) -> tuple:
    """
    Run single step of train data loader.
    Return loss, locations, and text.
    """
    locs, texts = batch

    if not isinstance(locs, torch.Tensor):
        raise TypeError(f"Expected locs to be a torch.Tensor, got {type(locs)}")

    if not isinstance(texts, (torch.Tensor, list, tuple)):
        raise TypeError(f"Expected texts to be a torch.Tensor, list, or tuple, got {type(texts)}")

    locs = locs.to(device)
    if isinstance(texts, torch.Tensor):
        texts = texts.to(device)

    text_features, location_features = model(texts, locs)

    loss = criterion(text_features, location_features)

    return loss, locs, texts

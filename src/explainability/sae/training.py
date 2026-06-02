import json
import torch.multiprocessing as mp
import os
from queue import Empty
from typing import Optional
from contextlib import nullcontext
from itertools import cycle

import torch as t
from tqdm import tqdm

import wandb

def get_norm_factor(data, steps: int, device) -> float:
    """Per Section 3.1, find a fixed scalar factor so activation vectors have unit mean squared norm.
    This is very helpful for hyperparameter transfer between different layers and models.
    Use more steps for more accurate results.
    https://arxiv.org/pdf/2408.05147
    
    If experiencing troubles with hyperparameter transfer between models, it may be worth instead normalizing to the square root of d_model.
    https://transformer-circuits.pub/2024/april-update/index.html#training-saes"""
    total_mean_squared_norm = 0
    count = 0

    for step, geo_location in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
        if step > steps:
            break
        
        count += 1
        mean_squared_norm = t.mean(t.sum(geo_location['point'].to(device) ** 2, dim=1))
        total_mean_squared_norm += mean_squared_norm

    average_mean_squared_norm = total_mean_squared_norm / count
    norm_factor = t.sqrt(average_mean_squared_norm).item()

    print(f"Average mean squared norm: {average_mean_squared_norm}")
    print(f"Norm factor: {norm_factor}")
    
    return norm_factor

def validation(geo_encoder,val_data, autocast_dtype, trainer, norm_factor=None, step=None, use_wandb=True):
    for use_threshold in [False, True]:
        l0s = []
        l2s = []
        fracs = []

        for geo_location in val_data:
            with t.no_grad():
                act = geo_encoder(geo_location['point'].to(trainer.device))
            act = act.detach().clone()
            act = act.to(dtype=autocast_dtype)
            act /= norm_factor
            with t.no_grad():
                f = trainer.ae.encode(act, use_threshold=use_threshold)
                act_hat = trainer.ae.decode(f)
                e = act - act_hat

            # Sparsity - L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()

            # Reconstruction - L2
            l2 = e.pow(2).sum(dim=-1).mean().item()

            # Reconstruction - Fraction of variance explained
            total_variance = t.var(act, dim=0).sum()
            residual_variance = t.var(e, dim=0).sum()
            frac_variance_explained = 1 - residual_variance / total_variance
            frac = frac_variance_explained.item()

            l0s.append(l0)
            l2s.append(l2)
            fracs.append(frac)

        threshold_str = "true" if use_threshold else "false"

        avg_frac_variance_explained = t.mean(t.tensor(fracs)).item()

        sparsity_log = {f"val_threshold_{threshold_str}/sparsity_l0": t.mean(t.tensor(l0s)).item()}
        reconstruction_log = {f"val_threshold_{threshold_str}/reconstruction_l2": t.mean(t.tensor(l2s)).item()}
        frac_variance_log = {f"val_threshold_{threshold_str}/frac_variance_explained": avg_frac_variance_explained}

        if use_wandb:
            if step is not None:
                wandb.log(sparsity_log, step=step)
                wandb.log(reconstruction_log, step=step)
                wandb.log(frac_variance_log, step=step)
            else:
                wandb.log(sparsity_log)
                wandb.log(reconstruction_log)
                wandb.log(frac_variance_log)
    
    return avg_frac_variance_explained

def trainSAE(
    geo_encoder,
    data,
    val_data,
    trainer_config: dict,
    steps: int,
    use_wandb: bool = True,
    save_steps:Optional[list[int]]=None,
    save_dir:Optional[str]=None,
    log_steps:Optional[int]=None,
    activations_split_by_head:bool=False,
    transcoder:bool=False,
    run_cfg:dict={},
    normalize_activations:bool=True,
    verbose:bool=False,
    device:str="cuda",
    autocast_dtype: t.dtype = t.float32,
):
    """
    Train SAEs using the given trainers

    If normalize_activations is True, the activations will be normalized to have unit mean squared norm.
    The autoencoders weights will be scaled before saving, so the activations don't need to be scaled during inference.
    This is very helpful for hyperparameter transfer between different layers and models.

    Setting autocast_dtype to t.bfloat16 provides a significant speedup with minimal change in performance.
    """ 

    device_type = "cuda" if "cuda" in device else "cpu"
    autocast_context = nullcontext() if device_type == "cpu" else t.autocast(device_type=device_type, dtype=autocast_dtype)
    
    trainer_class = trainer_config["trainer"]
    trainer_args = {k: v for k, v in trainer_config.items() if k != "trainer"}
    trainer = trainer_class(**trainer_args)

    # make save dir, export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        config_export = {"trainer": trainer.config}
        try:
            config_export["buffer"] = data.config
        except AttributeError:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_export, f, indent=4)
    
    best_val_frac_variance_explained = -float('inf') 
    best_model_path = os.path.join(save_dir, "best_ae.pt") if save_dir else None

    if normalize_activations:
        norm_factor = get_norm_factor(data, steps=100, device=device)
        trainer.config["norm_factor"] = norm_factor
        # Verify that all autoencoders have a scale_biases method
        trainer.ae.scale_biases(1.0)

    for step, geo_location in enumerate(tqdm(cycle(data), total=steps)):

        with t.no_grad():
            act = geo_encoder(geo_location['point'].to(device))
        act = act.detach().clone()  # TODO: maybe remove if activation dataset modified
        act = act.to(dtype=autocast_dtype)

        if normalize_activations:
            act /= norm_factor

        if step >= steps:
            break

        # validation / logging
        if log_steps is not None and step % log_steps == 0:
            current_frac_variance_explained = validation(geo_encoder,
                                                 val_data,
                                                 autocast_dtype,
                                                 trainer,
                                                 norm_factor=norm_factor,
                                                 step=step,
                                                 use_wandb=use_wandb)

            if current_frac_variance_explained > best_val_frac_variance_explained:
                best_val_frac_variance_explained = current_frac_variance_explained
                if save_dir is not None:
                    if normalize_activations:
                        trainer.ae.scale_biases(norm_factor)
                    checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
                    t.save(checkpoint, best_model_path)
                    if verbose:
                        print(f"New best model saved with frac variance explained: {best_val_frac_variance_explained:.4f}")
                    if normalize_activations:
                        trainer.ae.scale_biases(1 / norm_factor)

        # saving
        if save_steps is not None and step in save_steps and save_dir is not None:
            checkpoint_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            if normalize_activations:
                # Temporarily scale up biases for checkpoint saving
                trainer.ae.scale_biases(norm_factor)

            checkpoint = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
            t.save(
                checkpoint,
                os.path.join(checkpoint_dir, f"ae_{step}.pt"),
            )

            if normalize_activations:
                trainer.ae.scale_biases(1 / norm_factor)
        
        with autocast_context:
            loss_dict = trainer.update(step, act)
            if use_wandb:
                train_loss_log = {f"train/{k}": v for k, v in loss_dict.items()}
                wandb.log(train_loss_log, step=step)

    # save final SAE
    if save_dir is not None:
        if normalize_activations:
            trainer.ae.scale_biases(norm_factor)
        final = {k: v.cpu() for k, v in trainer.ae.state_dict().items()}
        t.save(final, os.path.join(save_dir, "ae.pt"))
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import os
import importlib
import warnings

import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from utils import get_location_model_output_dim
from modeling import LocationEncoder, TextEncoder
from modeling.model_specs import (
    format_default_model_specs,
    resolve_location_config,
    resolve_text_config,
)
from dataset import LocationDescriptionDataModule
from finetune import set_finetune_mode
from loss import ConceptLoss


class Location2TextLightningModule(lightning.pytorch.LightningModule):
    """Train a location encoder against a text encoder.

    Preferred args:
      - location_model: {backend, model_id, checkpoint}
      - text_model: {backend, model_id, pretrained, train_text_model, projection_*}
    """

    def __init__(
        self,
        location_model: Optional[dict[str, Any]] = None,
        text_model: Optional[dict[str, Any]] = None,
        hyperparameters: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        location_model = location_model or {}
        text_model = text_model or {}
        hyperparameters = hyperparameters or {}

        checkpoint_projection_only = hyperparameters.get("checkpoint_projection_only")
        finetune_mode = hyperparameters.get("finetune_mode")
        learning_rate = hyperparameters.get("learning_rate")
        weight_decay = hyperparameters.get("weight_decay")
        logit_scale_temperature = hyperparameters.get("logit_scale_temperature")
        logit_scale_max = hyperparameters.get("logit_scale_max")
        lambda_alignment = hyperparameters.get("lambda_alignment")
        sigma = hyperparameters.get("sigma")
        lr_scheduler = hyperparameters.get("lr_scheduler")

        location_backend = location_model.get("backend")
        location_model_id = location_model.get("model_id")
        location_checkpoint = location_model.get(
            "checkpoint", location_model.get("checkpoint_filename")
        )

        text_backend = text_model.get("backend")
        text_model_id = text_model.get("model_id")
        text_pretrained = text_model.get("pretrained")
        train_text_model = text_model.get("train_text_model")
        text_projection_head = text_model.get(
            "projection_layer", text_model.get("projection_head")
        )
        text_projection_hidden_dim = text_model.get("projection_hidden_dim")
        text_projection_dropout = text_model.get("projection_dropout")
        text_chunk_granularity = text_model.get("chunk_granularity")
        text_chunk_pooling = text_model.get("chunk_pooling")

        self.location_config = resolve_location_config(
            backend=location_backend,
            model_id=location_model_id,
            checkpoint_filename=location_checkpoint,
        )
        self.text_config = resolve_text_config(
            backend=text_backend,
            model_id=text_model_id,
            pretrained=text_pretrained,
        )

        # ---- encoders ----
        self.location_model = LocationEncoder(
            backend=self.location_config["backend"],
            model_id=self.location_config["model_id"],
            checkpoint_filename=self.location_config["checkpoint_filename"],
            train_location_model=False,
        )
        self.output_dim = get_location_model_output_dim(self.location_model)

        self.text_model = TextEncoder(
            backend=self.text_config["backend"],
            model_id=self.text_config["model_id"],
            pretrained=self.text_config["pretrained"],
            trainable=train_text_model,
            projection_head=text_projection_head,
            projection_hidden_dim=text_projection_hidden_dim,
            projection_dropout=text_projection_dropout,
            target_dim=self.output_dim,
            text_chunk_granularity=text_chunk_granularity,
            text_chunk_pooling=text_chunk_pooling,
        )
        self.checkpoint_projection_only = checkpoint_projection_only
        self.lambda_alignment = lambda_alignment

        # ---- loss / optimizer hyperparams ----
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler

        has_all_hparams = all(
            v is not None
            for v in [learning_rate, weight_decay, logit_scale_temperature,
                       lambda_alignment, sigma]
        )
        if has_all_hparams:
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / logit_scale_temperature),
                requires_grad=True,
            )
            self.loss_fn = ConceptLoss(
                logit_scale=self.logit_scale,
                logit_scale_max=logit_scale_max,
                lambda_alignment=lambda_alignment,
                sigma=sigma,
            )
            self.save_hyperparameters()
        else:
            self.logit_scale = None
            self.loss_fn = None

        set_finetune_mode(self.text_model, finetune_mode)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *args,
        **kwargs,
    ):
        if "strict" not in kwargs:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=kwargs.get("map_location", "cpu"),
            )
            state_dict = checkpoint.get("state_dict", {})
            is_projection_only = bool(state_dict) and all(
                k.startswith("text_model.embed_project.") for k in state_dict
            )

            hparams = checkpoint.get("hyper_parameters", {}) or {}
            text_cfg = hparams.get("text_model", {}) or {}
            text_backend = str(text_cfg.get("backend", "")).lower()
            is_gritlm = text_backend == "gritlm"

            if is_projection_only and is_gritlm:
                warnings.warn(
                    "Detected projection-only GritLM checkpoint; loading with strict=False.",
                    RuntimeWarning,
                )
                kwargs = dict(kwargs)
                kwargs["strict"] = False

        return super().load_from_checkpoint(checkpoint_path, *args, **kwargs)

    # ---- lifecycle hooks ----

    def on_fit_start(self):
        print(format_default_model_specs())
        print(f"Resolved location config: {self.location_config}")
        print(f"Resolved text config: {self.text_config}")
        if self.logger is not None:
            self.logger.log_hyperparams({
                "target_dim": self.output_dim,
                "text_dim": self.text_model.text_output_dim,
                "location_backend": self.location_config["backend"],
                "location_model_id": self.location_config["model_id"],
                "location_checkpoint": self.location_config["checkpoint_filename"],
                "text_backend": self.text_config["backend"],
                "text_model_id": self.text_config["model_id"],
                "text_pretrained": self.text_config["pretrained"],
            })

    # ---- forward / loss ----

    def forward_step(self, batch):
        locations, descriptions = batch
        text_emb = self.text_model(descriptions)
        loc_emb = self.location_model(locations)
        text_emb = text_emb / text_emb.norm(dim=1, keepdim=True)
        loc_emb = loc_emb / loc_emb.norm(dim=1, keepdim=True)
        return text_emb, loc_emb

    def _compute_loss(self, text_emb, loc_emb):
        if self.loss_fn is None:
            raise RuntimeError(
                "Loss not initialized. Provide all training hyperparameters "
                "or load from a checkpoint that includes them."
            )
        return self.loss_fn(text_emb, loc_emb)

    def training_step(self, batch):
        locations, _ = batch
        text_emb, loc_emb = self.forward_step(batch)
        loss = self._compute_loss(text_emb, loc_emb)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log_dict(
            {"train_loss": loss, "learning_rate": lr},
            on_step=True, on_epoch=True, prog_bar=True,
            batch_size=locations.size(0),
        )
        return loss

    def validation_step(self, batch):
        locations, _ = batch
        text_emb, loc_emb = self.forward_step(batch)
        loss = self._compute_loss(text_emb, loc_emb)
        self.log_dict(
            {"val_loss": loss},
            on_step=True, on_epoch=True,
            batch_size=locations.size(0),
        )
        return loss

    # ---- optimizer ----

    def configure_optimizers(self):
        if self.learning_rate is None or self.weight_decay is None:
            raise RuntimeError("Optimizer hyperparameters not set.")
        if self.logit_scale is None:
            raise RuntimeError("logit_scale not initialized.")

        self.logit_scale.requires_grad = True
        named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]

        def _no_decay(name, param):
            ln = name.lower()
            return param.ndim < 2 or any(k in ln for k in ("bn", "ln", "bias", "logit_scale"))

        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in named_params if _no_decay(n, p)], "weight_decay": 0.0},
                {"params": [p for n, p in named_params if not _no_decay(n, p)], "weight_decay": self.weight_decay},
            ],
            lr=self.learning_rate,
        )

        if self.lr_scheduler is None:
            return optimizer
        scheduler = self._build_lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    @staticmethod
    def _load_class(class_path: str):
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def _build_lr_scheduler(self, optimizer):
        scheduler_cfg = self.lr_scheduler
        if isinstance(scheduler_cfg, dict):
            class_path = scheduler_cfg.get("class_path")
            if not class_path:
                raise ValueError(
                    "lr_scheduler dict must include 'class_path'."
                )
            init_args = dict(scheduler_cfg.get("init_args", {}) or {})
            scheduler_cls = self._load_class(class_path)
            return scheduler_cls(optimizer=optimizer, **init_args)
        if isinstance(scheduler_cfg, type):
            return scheduler_cfg(optimizer=optimizer)
        if callable(scheduler_cfg):
            return scheduler_cfg(optimizer=optimizer)
        raise TypeError(
            "lr_scheduler must be either a dict with class_path/init_args, "
            "a scheduler class, or a callable that accepts optimizer."
        )

    # ---- inference helper ----

    @torch.no_grad()
    def text_model_predict(self, text, normalize=False):
        self.eval()
        emb = self.text_model(text)
        if normalize:
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    def on_save_checkpoint(self, checkpoint):
        if not self.checkpoint_projection_only:
            return

        state_dict = checkpoint.get("state_dict", {})
        keep_prefix = "text_model.embed_project."
        filtered = {k: v for k, v in state_dict.items() if k.startswith(keep_prefix)}
        if not filtered:
            raise RuntimeError(
                "checkpoint_projection_only=True but no projection-head weights were found "
                "under 'text_model.embed_project.*'."
            )

        checkpoint["state_dict"] = filtered
        # Drop trainer/runtime payload so files only contain projection weights.
        checkpoint.pop("optimizer_states", None)
        checkpoint.pop("lr_schedulers", None)
        checkpoint.pop("callbacks", None)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main(config_filename: str):
    lightning.pytorch.seed_everything(42)
    config_fn = Path(config_filename)

    cli = LightningCLI(
        model_class=Location2TextLightningModule,
        datamodule_class=LocationDescriptionDataModule,
        save_config_kwargs=dict(config_filename=config_fn, overwrite=True),
        trainer_defaults={"log_every_n_steps": 10},
        parser_kwargs={"default_config_files": [config_fn]},
        seed_everything_default=42,
        run=False,
    )

    for cb in cli.trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            cb.monitor = "val_loss"
            cb.patience = 5
            cb.mode = "min"
            cb.verbose = True
            cb.min_delta = 0.001
            break

    output_root = getattr(cli.datamodule, "output_root", None)
    if output_root:
        for cb in cli.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath:
                if not os.path.isabs(cb.dirpath):
                    cb.dirpath = os.path.join(output_root, cb.dirpath)

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lam = cli.model.lambda_alignment
    ds = cli.datamodule.hparams.dataset_name
    run_name = f"Location2Text_{ds}_lambda{lam}_{ts}"
    if cli.trainer.logger is not None:
        current_name = getattr(cli.trainer.logger.experiment, "name", None)
        # Keep explicit run names from config/sweep scripts.
        if current_name is None or str(current_name).strip() == "":
            cli.trainer.logger.experiment.name = run_name

    log_dir = cli.trainer.default_root_dir or "./lightning_logs"
    (Path(log_dir) / config_fn.parent).mkdir(parents=True, exist_ok=True)

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main("./configs/train.yaml")

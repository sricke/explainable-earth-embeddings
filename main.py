import sys
from pathlib import Path
from utils import get_location_model_output_dim
# Add project root to path so we can import satclip
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import lightning.pytorch
import torch
import numpy as np
import torch.nn as nn
from lightning.pytorch.cli import LightningCLI

from modeling import LocationEmbeddingModel, TextEmbeddingModel
from dataset import LocationDescriptionDataModule
from datetime import datetime
from finetune import set_finetune_mode

from loss import ConceptLoss

class Location2TextLightningModule(lightning.pytorch.LightningModule):
    def __init__(self, 
                 location_model_type: str,
                 location_model: str,
                 location_model_filename: str,
                 text_model_type: str,
                 text_model: str,
                 text_vocabulary: str,
                 train_text_model: bool,
                 finetune_mode: str,
                 learning_rate: float,
                 weight_decay: float,
                 logit_scale_temperature: float,
                 lambda_alignment: float,
                 sigma: float,
                 ):
        super().__init__()
        print('train_text_model', train_text_model)
        self.location_model = LocationEmbeddingModel(location_model_type=location_model_type, location_model=location_model, location_model_filename=location_model_filename, target_dim=None, train_location_model=False)
        self.output_dim = get_location_model_output_dim(self.location_model)

        self.text_model_str = text_model
        self.text_model = TextEmbeddingModel(text_model_type=text_model_type, text_model=text_model, text_vocabulary=text_vocabulary, train_text_model=train_text_model, target_dim=self.output_dim)


        self.learning_rate = learning_rate
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_temperature), requires_grad=True)
        self.logit_scale = logit_scale
        self.loss_fn = ConceptLoss(logit_scale=self.logit_scale, lambda_alignment=lambda_alignment, sigma=sigma)
        self.weight_decay = weight_decay
        self.save_hyperparameters()

        set_finetune_mode(self.text_model.model, finetune_mode)

    def _init_identity(self, layer: nn.Linear):
        """Initialize a square linear layer as identity + small noise."""
        assert layer.in_features == layer.out_features, "Must be square for identity init"
        nn.init.eye_(layer.weight)
        nn.init.zeros_(layer.bias)
        
    def on_fit_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams({'target_dim': self.output_dim,
                                         'text_dim': self.text_model.text_output_dim})
        
    def compute_loss(self, logits_per_text, logits_per_location):
        return self.loss_fn(logits_per_text, logits_per_location)
            
    def forward_step(self, batch):
        locations, descriptions = batch
        logits_per_text = self.text_model(descriptions)

        logits_per_location = self.location_model(locations)
        
        # normalize after projection
        logits_per_text = logits_per_text / logits_per_text.norm(dim=1, keepdim=True)
        logits_per_location = logits_per_location / logits_per_location.norm(dim=1, keepdim=True)

        return logits_per_text, logits_per_location

    def training_step(self, batch):
        locations, descriptions = batch
        logits_per_text, logits_per_location = self.forward_step(batch)
        loss = self.compute_loss(logits_per_text, logits_per_location)
        current_lr = self.optimizers().param_groups[0]['lr']
        batch_size = locations.size(0)
        self.log_dict(
            {"train_loss": loss, "learning_rate": current_lr},
            on_step=True, on_epoch=True, prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch):
        locations, descriptions = batch
        logits_per_text, logits_per_location = self.forward_step(batch)
        loss = self.compute_loss(logits_per_text, logits_per_location)
        self.log_dict(
            {"val_loss": loss},
            on_step=True, on_epoch=True,
            batch_size=locations.size(0),
        )
        return loss

    def configure_optimizers(self):
        # Collect all trainable parameters (as decided by set_finetune_mode)
        self.logit_scale.requires_grad = True
        named_params = [
            (name, p)
            for name, p in self.named_parameters()
            if p.requires_grad
        ]

        # Exclude biases, norms, and logit_scale from weight decay
        def exclude(name, p):
            lname = name.lower()
            return (
                p.ndim < 2
                or "bn" in lname
                or "ln" in lname
                or "bias" in lname
                or "logit_scale" in name
            )

        no_decay_params = [p for name, p in named_params if exclude(name, p)]
        decay_params = [p for name, p in named_params if not exclude(name, p)]

        optimizer = torch.optim.AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": self.weight_decay},
            ],
            lr=self.learning_rate,
        )
        return optimizer

    def text_model_predict(self, text, normalize=False):
        self.eval()
        with torch.no_grad():
            # text_model will tokenize internally
            embeddings = self.text_model(text)
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

from lightning.pytorch.callbacks import EarlyStopping
        
def cli_main(config_filename: str):

    lightning.pytorch.seed_everything(42)

    config_fn = Path(config_filename)
   
    cli = LightningCLI(
        model_class=Location2TextLightningModule,
        datamodule_class=LocationDescriptionDataModule,
        save_config_kwargs=dict(
            config_filename=config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "log_every_n_steps": 10
        },
        parser_kwargs={"default_config_files": [config_fn]},
        seed_everything_default=42,
        run=False,
    )
    early_stop_cb = None
    for cb in cli.trainer.callbacks:
        if isinstance(cb, EarlyStopping):
            early_stop_cb = cb
            break

    # now modify it (ensure EarlyStopping is configured as desired)
    if early_stop_cb is not None:
        early_stop_cb.monitor = "val_loss"
        early_stop_cb.patience = 5
        early_stop_cb.mode = "min"
        early_stop_cb.verbose = True
        early_stop_cb.min_delta = 0.001

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lambda_align = cli.model.hparams.lambda_alignment
    dataset_name = cli.datamodule.hparams.dataset_name
    run_name = f"Location2Text_{dataset_name}_S2_lambda{lambda_align}_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name

    
    # Create folder to log configs
    dirname_cfg = Path(config_fn).parent
    log_dir = cli.trainer.default_root_dir or "./lightning_logs"
    dir_log_cfg = Path(log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)
    

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    config_fn = "./configs/train.yaml"

    cli_main(config_fn)
        

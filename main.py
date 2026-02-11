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
import open_clip

from loss import ConceptLoss

class Location2TextLightningModule(lightning.pytorch.LightningModule):
    def __init__(self, 
                 location_model: str,
                 location_model_filename: str,
                 text_model: str,
                 text_vocabulary: str,
                 train_text_model: bool,
                 learning_rate: float,
                 weight_decay: float,
                 logit_scale_temperature: float,
                 lambda_alignment: float,
                 sigma: float,
                 ):
        super().__init__()
        print('train_text_model', train_text_model)
        self.location_model = LocationEmbeddingModel(location_model=location_model, location_model_filename=location_model_filename, target_dim=None, train_location_model=False)
        self.output_dim = get_location_model_output_dim(self.location_model)
        self.text_model_str = text_model
        self.text_model = TextEmbeddingModel(text_model=text_model, text_vocabulary=text_vocabulary, train_text_model=train_text_model, target_dim=self.output_dim)
        self.learning_rate = learning_rate
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / logit_scale_temperature), requires_grad=True)
        self.loss_fn = ConceptLoss(logit_scale=logit_scale, lambda_alignment=lambda_alignment, sigma=sigma)
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        
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
        logits_per_text, logits_per_location = self.forward_step(batch)
        loss = self.compute_loss(logits_per_text, logits_per_location)
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log_dict({"train_loss": loss,
                  "learning_rate": current_lr}, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch):
        logits_per_text, logits_per_location = self.forward_step(batch)
        loss = self.compute_loss(logits_per_text, logits_per_location)
        self.log_dict({"val_loss": loss}, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        named_params = []
    
        if self.text_model.train_encoder:
            named_params.extend([
                (f"text_model.{n}", p) 
                for n, p in self.text_model.named_parameters() 
                if p.requires_grad
            ])
        
        named_params.append(("loss_fn.logit_scale", self.loss_fn.logit_scale))
        
        # no decay for biases, norms, and 1D params
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        
        no_decay_params = [p for n, p in named_params if exclude(n, p)]
        decay_params = [p for n, p in named_params if not exclude(n, p)]
        
        optimizer = torch.optim.AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0.0},
                {"params": decay_params, "weight_decay": self.weight_decay},
            ],
            lr=self.learning_rate,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update each epoch
            }
        }

    def text_model_predict(self, text, normalize=False):
        self.eval()
        with torch.no_grad():
            # text_model will tokenize internally
            embeddings = self.text_model(text)
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings


        
def cli_main(config_filename: str):
    config_fn = Path(config_filename)
   
    cli = LightningCLI(
        model_class=Location2TextLightningModule,
        datamodule_class=LocationDescriptionDataModule,
        save_config_kwargs=dict(
            config_filename=config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "log_every_n_steps": 10 # controls how often Lightning sends metrics to loggers
        },
        parser_kwargs={"default_config_files": [config_fn]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    lambda_align = cli.model.hparams.lambda_alignment
    run_name = f"Location2Text_S2_lambda{lambda_align}_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name

    
    # Create folder to log configs
    dirname_cfg = Path(config_fn).parent
    dir_log_cfg = Path(cli.trainer.log_dir) / dirname_cfg
    dir_log_cfg.mkdir(parents=True, exist_ok=True)
    

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    config_fn = "./configs/train.yaml"

    cli_main(config_fn)
        

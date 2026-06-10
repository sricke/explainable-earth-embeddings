import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime

import torch
#print(torch.__version__)
from eval_data import CoordinatesDataModule
from location_encoders.location_encoder import _load_model, LOCATION_EMBEDDING_DIMENSIONS
from location_encoders.satclip.datamodules.s2geo_dataset import S2GeoDataModule

from saes import BatchTopKSAE, BatchTopKTrainer
from training import trainSAE
import wandb
wandb.login()
#from util import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataset(dataset_name, data_dir):
    if dataset_name == "S2100k":
        dataset = S2GeoDataModule(data_dir=data_dir, mode="points", batch_size=1024, transform=None)
    else:
        dataset = CoordinatesDataModule(data_dir=data_dir, batch_size=1024, num_workers=6)
    dataset.setup()
    return dataset


if __name__ == "__main__":

    location_encoder_name = "satclip"
    location_encoder_model = _load_model(location_encoder_name, device)
    embedding_dim = LOCATION_EMBEDDING_DIMENSIONS[location_encoder_name]
    print(f"loaded {location_encoder_name} location encoder with embedding dimension {embedding_dim} on device {device}")


    #uar datasets
    dataset_name= "urban_geoyfcc"
    dataset = load_dataset(dataset_name, "/home/datasets/earth_embeddings/{}/".format(dataset_name))
    train_loader = dataset.train_dataloader()
    validation_loader = dataset.val_dataloader()
    steps = 10000
    trainer_cfg = {
        "trainer": BatchTopKTrainer,
        "activation_dim": embedding_dim,
        "dict_size": embedding_dim,
        "k": 20,
        "device": device,
        "steps": steps,
        "layer": "",
        "lm_name": location_encoder_name,
        "submodule_name": "",
        "lr": 3e-4}


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(project='SAE_{}_{}'.format(location_encoder_name, dataset_name), config={timestamp: timestamp})
    save_dir = "/home/results/sae_geo_embeddings/{}/{}/{}/dict_size={},k={}/{}".format(dataset_name, trainer_cfg["lm_name"], trainer_cfg["trainer"].__name__, trainer_cfg["dict_size"], trainer_cfg["k"], timestamp)
    trainSAE(location_encoder_model, train_loader, validation_loader, trainer_cfg, steps, use_wandb=True, log_steps=175, save_steps=[1000, 5000, 9999, 14999], save_dir=save_dir)

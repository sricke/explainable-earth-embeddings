from collections.abc import Callable
import os

from matplotlib.pyplot import Figure
from torch import Tensor

from torchgeo.datasets import NonGeoDataset
from torchgeo.datasets.utils import Path

import random
import pandas as pd

import lightning.pytorch as pl
from torch.utils.data import DataLoader

import open_clip
import torch

class LocationDescriptionDataset(NonGeoDataset):
    """
        
    
    Dataset format:

        * fn: str -> patch id
        * lat: float
        * lon: float
        * description: str

    """

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train', # train, val, test
        text_model: str = 'ViT-B-32',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize the dataset.

        As of now, we are only interested in (lat, lon) and labels and not actually dealing with image transforms and bands

        Args:
            root: root directory where dataset can be found.Dataset should be in csv format with columns: fn, lat, lon, description
            split: one of "train", "val", or "test" 
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.data = pd.read_csv(f"{root}/{split}.csv")
        self.transforms = transforms

    def __len__(self) -> int:
        """The length of the dataset"""
        return len(self.data)
        

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """A single sample from the dataset.

        Load a single input image and target label or mask, and return it in a
        dictionary.
        """
        if self.transforms is not None:
            raise NotImplementedError("Transformations are not implemented for this dataset yet")
        row = self.data.iloc[index]
        label = row['description']
        point = torch.tensor([row['lon'], row['lat']])
        tokenized_label = self.tokenizer(label).squeeze(0)
        return point, tokenized_label

    def plot(self) -> Figure:
        """Plot a sample of the dataset for visualization purposes.
        """
        raise NotImplementedError("Plotting is not implemented for this dataset yet")

class LocationDescriptionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        dataset_name: str = "default",
        batch_size: int = 64,
        num_workers: int = 6,
        text_model: str = 'ViT-B-32',
        transform: str = None,
        mode: str = "both",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.text_model = text_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform
        if self.train_transform is not None:
            raise NotImplementedError("Transformations are not implemented for this dataset yet")

        self.mode = mode
        self.save_hyperparameters()
        
        self.columns = ['fn', 'lat', 'lon', 'description']

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No dataset found. Data path {self.data_path} does not exist")
        df = pd.read_csv(f'{self.data_path}/train.csv')
        if df.empty:
            raise ValueError(f"Data path {self.data_path} is empty")
        for column in self.columns:
            if column not in df.columns:
                raise ValueError(f"Data path {self.data_path} does not contain {column} column")

    def setup(self, stage="fit"):
        self.train_dataset = LocationDescriptionDataset(root=self.data_path, split='train', text_model=self.text_model, transforms=self.train_transform)
        self.val_dataset = LocationDescriptionDataset(root=self.data_path, split='val', text_model=self.text_model, transforms=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            #persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader is not implemented for this dataset yet")

        
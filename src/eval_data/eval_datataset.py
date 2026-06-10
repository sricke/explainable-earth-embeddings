import os
from typing import Any, Callable, Dict, Optional

import pandas as pd
import rasterio
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np
import torch

import lightning.pytorch as pl
from torch.utils.data import DataLoader

CHECK_MIN_FILESIZE = 10000 # 10kb

class CoordinatesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/data/geoclip_s2",
        batch_size: int = 64,
        num_workers: int = 6,
        val_random_split_fraction: float = 0.1
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
            
        self.val_random_split_fraction = val_random_split_fraction
        self.save_hyperparameters()


    def setup(self, stage="fit"):
        dataset = PointDataset(root=self.data_dir)

        generator = torch.Generator().manual_seed(42)
        N_val = int(len(dataset) * self.val_random_split_fraction)
        N_train = len(dataset) - N_val
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [N_train, N_val], generator=generator)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
        raise NotImplementedError

class PointDataset(Dataset):
    """Point dataset.

    This dataset contains 100,000 256x256 patches of 12 band Sentinel imagery sampled randomly
    from Sentinel 2 scenes on the Microsoft Planetary Computer that have <20% cloud cover,
    intersect land, and were captured between 2021-01-01 and 2023-05-17 (there are 2,359,972
    such scenes).
    """

    def __init__(
        self,
        root: str,
    ) -> None:
        """Initialize a new Point dataset instance.
        Args:
            root: root directory of S2-100K pre-sampled dataset
            transform: torch transform to apply to a sample
            mode: which data to return (options are "both" or "points"), useful for embedding locations without loading images 
        """
        self.root = root

        index_fn = "index.csv"

        df = pd.read_csv(os.path.join(self.root, index_fn))
        self.filenames = []
        self.points = []

        n_skipped_files = 0
        for i in range(df.shape[0]):
            self.points.append((df.iloc[i]["lon"], df.iloc[i]["lat"]))


    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            dictionary with "image" and "point" keys where point is in (lon, lat) format
        """
        point = torch.tensor(self.points[index])
        sample = {"point": point}
            
        return sample

    def __len__(self) -> int:
        """Return the number of datapoints in the dataset.
        Returns:
            length of dataset
        """
        return len(self.points)
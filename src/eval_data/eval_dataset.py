import os
from os.path import exists
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image as im
import pandas as pd
import rasterio
from tqdm import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


import lightning.pytorch as pl
from torch.utils.data import DataLoader

CHECK_MIN_FILESIZE = 10000 # 10kb

class CoordinatesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",  # insert path to geoclip S2 data
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
            shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError

class PointDataset(Dataset):
    """Point dataset.

    A dataset class for handling point data.
    """

    def __init__(
        self,
        root: str,
    ) -> None:
        """Initialize a new Point dataset instance.
        Args:
            root: root directory of the sampled dataset
            transform: torch transform to apply to a sample
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

# Code extracted from https://github.com/VicenteVivan/geo-clip/blob/main/geoclip/train/dataloader.py

def geo_clip_img_train_transform():
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform_list

def geo_clip_img_val_transform():
    val_transform_list = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return val_transform_list    


class GeoClipImageryDataset(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        returned_files = 0
        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = row['IMG_FILE']
            if exists(filename):
                images.append(filename)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))
                returned_files += 1

        print(f"Loaded {returned_files}/{len(dataset_info)} entries from {dataset_file}")

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = torch.tensor(self.coordinates[idx])
        image = im.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        sample = {"point": gps, "image": image, "filename": img_path}
        return sample

if __name__ == "__main__":
    # 1. Load your original dataset
    # Replace 'your_dataset.csv' with the actual file name
    df = pd.read_csv('/home/datasets/earth_embeddings/geoclip_imagery/yfcc_row_mapping.csv')

    # 2. Sample 10% of the rows randomly
    # frac=0.1 specifies 10%, random_state ensures reproducibility
    sampled_df = df.sample(frac=0.1, random_state=42)

    # 3. Save the sampled data to a new file
    sampled_df.to_csv('/home/datasets/earth_embeddings/geoclip_imagery/sampled_index.csv', index=False)

    print(f"Successfully saved 10% of data ({len(sampled_df)} rows) to sampled_index.csv")
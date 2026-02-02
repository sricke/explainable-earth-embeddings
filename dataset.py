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

class BigEarthNetv2_S2Dataset(NonGeoDataset):
    """BigEarthNet-S2 dataset.
    
    https://bigearth.net/
    
    This class is for training location encoders on the BigEarthNet v2.0 dataset. As such, it has 
    additional information of lat, lon of individual patches not present in the original dataset.
    It only includes images from Sentinel-2.

    BigEarthNet v2.0 is a benchmark dataset consisting of 549,488 pairs of Sentinel-1 and Sentinel-2
    image patches. To construct BigEarthNet v2.0 with Sentinel-2 image patches (called as BigEarthNet-
    S2), 115 Sentinel-2 tiles acquired between June 2017 and May 2018 over 10 countries (Austria,
    Belgium, Finland, Ireland, Kosovo, Lithuania, Luxembourg, Portugal, Serbia, and Switzerland) of
    Europe were initially selected. All the tiles were atmospherically corrected by the Sentinel-2 Level 2A
    product generation and formatting tool (sen2cor v2.11). Then, they were divided into 549,488 image
    patches. Each image patch was associated with a pixel-level reference map and multiple land-cover
    class labels (i.e., multi-labels) that were derived from the most recent CORINE Land Cover database
    of the year 2018 (CLC2018 v2020_u1).
    
    BigEarthNet v2.0 introduces a new geographical-based split assignment algorithm, which significantly
    reduces spatial correlation among the train, validation, and test sets compared to v1.0.
    
    The original dataset is divided between no snow and cloud coverage (<1%). 
    
    Classes present:
        - Agro-forestry areas
        - Arable land
        - Beaches, dunes, sands
        - Broad-leaved forest
        - Coastal wetlands
        - Complex cultivation patterns
        - Coniferous forest
        - Industrial or commercial units
        - Inland waters
        - Inland wetlands
        - Land principally occupied by agriculture, with significant areas of natural vegetation
        - Marine waters
        - Mixed forest
        - Moors, heathland and sclerophyllous vegetation
        
    
    Dataset format:

        * pathc_id: str
        <Sentinel-ID>_MSIL2A_<YYYYMMDD>T<HHMMSS>_N9999_<Rooo>_<Txxxxxx>_<H-Order>_<V-Order>
            Txxxxxx: Tile ID
            H-Order: Horizontal order
            V-Order: Vertical order
        * labels: list[str]
        * country: str
        * lat: float
        * lon: float

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
            root: root directory where dataset can be found.Dataset should be in parquet format with columns: patch_id, labels, country, lat, lon
            split: one of "train", "val", or "test" 
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
        """
        self.tokenizer = open_clip.get_tokenizer(text_model)
        self.data = pd.read_parquet(f"{root}/{split}.parquet")
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
        label = random.choice(row['labels'])
        tokenized_label = self.tokenizer(label).squeeze(0)
        point = torch.tensor([row['lon'], row['lat']])
        return point, tokenized_label

    def plot(self) -> Figure:
        """Plot a sample of the dataset for visualization purposes.
        """
        raise NotImplementedError("Plotting is not implemented for this dataset yet")

class BigEarthNetv2_S2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 64,
        num_workers: int = 6,
        text_model: str = 'ViT-B-32',
        transform: str = None,
        mode: str = "both",
    ):
        super().__init__()
        self.data_path = data_path
        self.text_model = text_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform
        if self.train_transform is not None:
            raise NotImplementedError("Transformations are not implemented for this dataset yet")

        self.mode = mode
        self.save_hyperparameters()
        
        self.columns = ['lat', 'lon', 'labels']

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No dataset found. Data path {self.data_path} does not exist")
        df = pd.read_parquet(self.data_path)
        if df.empty:
            raise ValueError(f"Data path {self.data_path} is empty")
        for column in self.columns:
            if column not in df.columns:
                raise ValueError(f"Data path {self.data_path} does not contain {column} column")

    def setup(self, stage="fit"):
        self.train_dataset = BigEarthNetv2_S2Dataset(root=self.data_path, split='train', text_model=self.text_model, transforms=self.train_transform)
        self.val_dataset = BigEarthNetv2_S2Dataset(root=self.data_path, split='val', text_model=self.text_model, transforms=None)

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
        raise NotImplementedError("Test dataloader is not implemented for this dataset yet")

        
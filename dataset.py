from collections.abc import Callable
import os
from pathlib import Path
from typing import Optional

from matplotlib.pyplot import Figure
from torch import Tensor

from torchgeo.datasets import NonGeoDataset

import random
import pandas as pd

import lightning.pytorch as pl
from torch.utils.data import DataLoader

import open_clip
import torch
from datasets.geoyfcc import GeoYFCCTextDataset
from datasets.wildsat import WildSatTextDataset

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
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        download: bool = False,
        debug_print_examples: int = 0,
        text_embedding_cache_path: Optional[str] = None,
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
        self.data = pd.read_csv(f"{root}/{split}.csv")
        self.transforms = transforms
        self.debug_print_examples = max(0, int(debug_print_examples))
        self.text_column: Optional[str] = None
        if "description" in self.data.columns:
            self.text_column = "description"
        elif "text" in self.data.columns:
            self.text_column = "text"
        self.embedding_index_column = "embedding_index"
        if text_embedding_cache_path is None and self.text_column is None:
            raise ValueError(
                f"{root}/{split}.csv must contain either 'description' or 'text' column"
            )
        self.cached_text_embeddings: Optional[torch.Tensor] = None
        if text_embedding_cache_path is not None:
            cache_path = Path(text_embedding_cache_path).expanduser()
            if not cache_path.exists():
                raise FileNotFoundError(f"Text embedding cache not found: {cache_path}")
            cached = torch.load(cache_path, map_location="cpu")
            if not isinstance(cached, torch.Tensor):
                raise TypeError(
                    f"Expected a torch.Tensor cache at {cache_path}, got {type(cached)!r}"
                )
            if cached.ndim != 2:
                raise ValueError(
                    f"Cached embeddings must be 2D [N, D], got shape={tuple(cached.shape)}"
                )
            if self.embedding_index_column not in self.data.columns:
                raise ValueError(
                    f"{root}/{split}.csv must contain '{self.embedding_index_column}' "
                    "column when using text_embedding_cache_path."
                )
            idx = self.data[self.embedding_index_column]
            if idx.isna().any():
                raise ValueError(
                    f"Found NaN values in '{self.embedding_index_column}' "
                    f"for split={split!r}."
                )
            min_idx = int(idx.min())
            max_idx = int(idx.max())
            if min_idx < 0 or max_idx >= int(cached.shape[0]):
                raise ValueError(
                    f"Embedding indices out of bounds for split={split!r}: "
                    f"min={min_idx}, max={max_idx}, cache_rows={cached.shape[0]}"
                )
            self.cached_text_embeddings = cached

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
        label: str | torch.Tensor
        if self.cached_text_embeddings is not None:
            emb_idx = int(row[self.embedding_index_column])
            label = self.cached_text_embeddings[emb_idx]
        else:
            if self.text_column is None:
                raise ValueError(
                    "No text column available ('description' or 'text') "
                    "and no text embedding cache configured."
                )
            label = row[self.text_column]
            if isinstance(label, (tuple, list)):
                label = list(label)
            else:
                label = str(label)

            # Optional debug print to inspect raw text going to the encoder.
            if not hasattr(self, "_debug_printed_examples"):
                self._debug_printed_examples = 0
            if self._debug_printed_examples < self.debug_print_examples:
                snippet = label[:300]
                print(
                    "[LocationDescriptionDataset.__getitem__] "
                    f"index={index}, text_column={self.text_column!r}, "
                    f"text_len={len(label)} chars, "
                    f"snippet={snippet!r}, lat={row['lat']}, lon={row['lon']}"
                )
                self._debug_printed_examples += 1

        point = torch.tensor([row['lat'], row['lon']], dtype=torch.float32) #CHANGE FOR SATCLIP
        return point, label

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
        transform: str = None,
        mode: str = "both",
        source_csv_path: str | None = None,
        source_pkl_path: str | None = None,
        test_size: float = 0.1,
        random_state: int = 42,
        debug_print_examples: int = 0,
        text_embedding_cache_path: str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform
        if self.train_transform is not None:
            raise NotImplementedError("Transformations are not implemented for this dataset yet")

        self.mode = mode
        self.source_csv_path = source_csv_path
        self.source_pkl_path = source_pkl_path
        self.test_size = test_size
        self.random_state = random_state
        self.debug_print_examples = max(0, int(debug_print_examples))
        self.text_embedding_cache_path = text_embedding_cache_path
        self.save_hyperparameters()

        self.columns = ['lat', 'lon', 'text']

    def _split_paths(self) -> tuple[str, str]:
        train_csv = os.path.join(self.data_path, "train.csv")
        val_csv = os.path.join(self.data_path, "val.csv")
        return train_csv, val_csv

    def _ensure_splits_exist(self) -> None:
        train_csv, val_csv = self._split_paths()
        if os.path.exists(train_csv) and os.path.exists(val_csv):
            print(
                "[LocationDescriptionDataModule._ensure_splits_exist] "
                f"Found existing splits: train={train_csv}, val={val_csv}"
            )
            return

        dataset_name = str(self.dataset_name).lower()
        if "wildsat" in dataset_name:
            print(
                "[LocationDescriptionDataModule._ensure_splits_exist] "
                f"dataset_name={self.dataset_name!r} → using WildSatTextDataset path"
            )
            raw_csv_path = self.source_csv_path
            if raw_csv_path is None:
                candidate = os.path.join(self.data_path, "geolocated_text_dataset.csv")
                if os.path.exists(candidate):
                    raw_csv_path = candidate
            if raw_csv_path is None:
                raise FileNotFoundError(
                    "WildSat splits are missing and no `source_csv_path` was provided. "
                    "Pass the raw WildSat CSV path (e.g. geolocated_text_dataset.csv)."
                )
            print(
                "[LocationDescriptionDataModule._ensure_splits_exist] "
                f"Creating WildSat splits from raw_csv_path={raw_csv_path} "
                f"into data_path={self.data_path}"
            )
            WildSatTextDataset.create_splits_from_wildsat_csv(
                wildsat_csv_path=Path(raw_csv_path),
                output_dir=Path(self.data_path),
                test_size=self.test_size,
                random_state=self.random_state,
            )
            return

        if "geoyfcc" in dataset_name:
            print(
                "[LocationDescriptionDataModule._ensure_splits_exist] "
                f"dataset_name={self.dataset_name!r} → using GeoYFCCTextDataset path"
            )
            pkl_path = self.source_pkl_path
            if pkl_path is None:
                pkl_path = "~/data/geoyfcc/geoyfcc_text_filtered_single_label.pkl"
            print(
                "[LocationDescriptionDataModule._ensure_splits_exist] "
                f"Creating GeoYFCC splits from pkl_path={pkl_path} "
                f"into data_path={self.data_path}"
            )
            GeoYFCCTextDataset.create_splits_from_pickle(
                pkl_path=Path(pkl_path),
                output_dir=Path(self.data_path),
                test_size=self.test_size,
                random_state=self.random_state,
            )
            return

        if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
            raise FileNotFoundError(
                "Missing train/val CSV splits. Provide precomputed train.csv and val.csv "
                "in `data_path`, or set `dataset_name` to include 'wildsat'/'geoyfcc' "
                "with the corresponding source path arguments."
            )

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"No dataset found. Data path {self.data_path} does not exist")
        self._ensure_splits_exist()
        train_csv, _ = self._split_paths()
        print(
            f"[LocationDescriptionDataModule.prepare_data] "
            f"Reading train split from {train_csv}"
        )
        df = pd.read_csv(train_csv)
        if df.empty:
            raise ValueError(f"Data path {self.data_path} is empty")
        if self.text_embedding_cache_path is not None:
            columns = ['lat', 'lon', 'embedding_index']
        else:
            columns = ['lat', 'lon', 'description'] if 'description' in df.columns else ['lat', 'lon', 'text']
        self.columns = columns
        for column in columns:
            if column not in df.columns:
                raise ValueError(f"Data path {self.data_path} does not contain {column} column")

    def setup(self, stage="fit"):
        self.train_dataset = LocationDescriptionDataset(
            root=self.data_path,
            split='train',
            transforms=self.train_transform,
            debug_print_examples=self.debug_print_examples,
            text_embedding_cache_path=self.text_embedding_cache_path,
        )
        self.val_dataset = LocationDescriptionDataset(
            root=self.data_path,
            split='val',
            transforms=None,
            debug_print_examples=self.debug_print_examples,
            text_embedding_cache_path=self.text_embedding_cache_path,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader is not implemented for this dataset yet")
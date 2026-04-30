import logging
import os
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import Figure
from utils import set_seed

logger = logging.getLogger(__name__)

class GeoTextDataset(Dataset):
    """
    Dataset consisting of latlon pairs with text description.

    Currently supports csv format, with columns lat, lon, text
    """

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train', # train, val, test
        subset_size: int = None, 
        precomputed_text_embeddings: bool = True,
        precomputed_location_embeddings: bool = True
    ) -> None:
        
        super().__init__()

        self.root = Path(root).expanduser()

        parquet_path = self.root / f"{split}.parquet"
        csv_path = self.root / f"{split}.csv"

        data_parquet = self.root / "data.parquet"
        data_csv = self.root / "data.csv"

        if parquet_path.exists() or csv_path.exists():
            if parquet_path.exists():
                self.df = pd.read_parquet(parquet_path)
            else:
                self.df = pd.read_csv(csv_path, index_col=False)

        else:
            if data_parquet.exists():
                df = pd.read_parquet(data_parquet)
                fmt = "parquet"
            elif data_csv.exists():
                df = pd.read_csv(data_csv, index_col=False)
                fmt = "csv"
            else:
                raise FileNotFoundError(f"Neither split nor data.parquet/csv exists in {data_parquet} or {data_csv}")

            self._save_train_test_split(df, fmt=fmt)

            split_path = self.root / f"{split}.{fmt}"
            self.df = pd.read_parquet(split_path) if fmt == "parquet" else pd.read_csv(split_path, index_col=False)

        if subset_size is not None:
            self.subsample(subset_size)

        self.precomputed_text_embeddings = precomputed_text_embeddings and 'text_embedding' in self.df.columns
        self.precomputed_location_embeddings = precomputed_location_embeddings and 'location_embedding' in self.df.columns

        if not self.precomputed_text_embeddings:
            assert 'text' in self.df.columns or 'caption' in self.df.columns, "Data should have text or caption column"
        if not self.precomputed_location_embeddings:
            assert all(col in self.df.columns for col in ['lat', 'lon']), f"Data csv does not contain lat lon columns"

    def __len__(self) -> int:
        """The length of the dataset"""
        return len(self.df)

    def __getitem__(self, index: int, return_dict=False):
        row = self.df.iloc[index]

        if self.precomputed_location_embeddings:
            loc = row["location_embedding"]
            loc_key = "location_embedding"
        else:
            loc = torch.tensor([row["lat"], row["lon"]], dtype=torch.float32)
            loc_key = "latlon"

        if self.precomputed_text_embeddings:
            txt = row["text_embedding"]
            txt_key = "text_embedding"
        else:
            txt_col = "text" if "text" in self.df.columns else "caption"
            txt = row[txt_col]
            txt_key = "text"

        # Ensure precomputed embeddings collate into proper tensors 
        if self.precomputed_location_embeddings:
            loc = torch.tensor(loc, dtype=torch.float32)
        if self.precomputed_text_embeddings:
            txt = torch.tensor(txt, dtype=torch.float32)

        if return_dict:
            return {
                loc_key: loc,
                txt_key: txt,
            }

        return loc, txt
    
    def subsample(self, n: int, seed: int = 42):
        self.df = self.df.sample(n=n, random_state=seed).reset_index(drop=True)

    
    def _save_train_test_split(self, all_data: pd.DataFrame = None, val_size=0.1, test_size=0.1, random_state=42, fmt="csv"):
        from sklearn.model_selection import train_test_split

        assert all_data is not None, "Need data df"

        train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
        train_data, val_data = train_test_split(train_data, test_size=val_size, random_state=42)

        for split_name, df in {'train': train_data, 'val': val_data, 'test': test_data}.items():
            path = Path(self.root) / f"{split_name}.{fmt}"
            df.to_parquet(path, index=False) if fmt == "parquet" else df.to_csv(path, index=False)
            print(f"Saved {split_name} split: {path} ({len(df)} rows)")

        return train_data, val_data, test_data



    def plot(self, color=None, s=5) -> Figure:
        " Plot of lat lon points"
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        import numpy as np

        title = "Plot of data points"

        fig, ax = plt.subplots(1, figsize=(6, 3))

        m = Basemap(projection='cyl', resolution='c', ax=ax)
        m.drawcoastlines()

        coords = self.df[['lon', 'lat']].to_numpy()  # lon first
        ax.scatter(coords[:,0], coords[:,1], c=color, s=s, alpha=0.7)
        ax.set_title(title)

        return fig


def build_dataloaders(
    dataset_path,
    batch_size,
    num_workers,
    seed,
    precomputed_text_embeddings=True,
    precomputed_location_embeddings=True,
    train_subsample_size=None,
    val_subsample_size=None,
):
    train_dataset = GeoTextDataset(
        root=dataset_path,
        split='train',
        subset_size=train_subsample_size,
        precomputed_text_embeddings=precomputed_text_embeddings,
        precomputed_location_embeddings=precomputed_location_embeddings,
    )
    val_dataset = GeoTextDataset(
        root=dataset_path,
        split='val',
        subset_size=val_subsample_size,
        precomputed_text_embeddings=precomputed_text_embeddings,
        precomputed_location_embeddings=precomputed_location_embeddings,
    )
    g = torch.Generator()
    g.manual_seed(seed)
    worker_init = lambda wid: set_seed(seed + wid)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=worker_init, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init)
    logger.info(f"Loaded dataset from {dataset_path}")
    logger.info(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}")
    return train_dataloader, val_dataloader
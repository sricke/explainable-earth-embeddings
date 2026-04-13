import os
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset

from matplotlib.pyplot import Figure

class GeoTextDataset(Dataset):
    """
    Dataset consisting of latlon pairs with text description.

    Currently supports csv format, with columns lat, lon, text
    """

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train', # train, val, test
        precomputed_text_embeddings: bool = True,
        precomputed_location_embeddings: bool = True
    ) -> None:
        
        super().__init__()

        self.root = root

        all_data_csv_path = Path(root) / "data.csv"
        split_csv_path = Path(root) / f"{split}.csv"

        if os.path.exists(all_data_csv_path) and not os.path.exists(split_csv_path):
            all_data = pd.read_csv(all_data_csv_path, index_col=False)
            self._save_train_test_split(all_data)

        split_parquet_path = Path(root) / f"{split}.parquet"

        if os.path.exists(split_parquet_path):
            self.df = pd.read_parquet(split_parquet_path)
            precomputed_text_embeddings = precomputed_location_embeddings = True
        else:
            self.df = pd.read_csv(split_csv_path, index_col=False)

        self.precomputed_text_embeddings = precomputed_text_embeddings
        self.precomputed_location_embeddings = precomputed_location_embeddings

        if precomputed_text_embeddings:
            assert 'text_embedding' in self.df.columns, "Data should have text_embedding column"
        else:
            assert 'text' in self.df.columns, "Data should have text column"
        if precomputed_location_embeddings:
            assert 'location_embedding' in self.df.columns, "Data should have location_embedding column"
        else:
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
            txt = row["text"]
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
    
    def _save_train_test_split(self, all_data: pd.DataFrame = None, val_size=0.1, test_size=0.1, random_state=42):
        from sklearn.model_selection import train_test_split

        assert all_data is not None, "Need data df"

        train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
        train_data, val_data = train_test_split(train_data, val_size=val_size, random_state=42)

        splits = {'train': train_data, 'val': val_data, 'test': test_data}

        for split_name, df in splits.items():
            split_csv_path = Path(self.root) / f"{split_name}.csv"
            df.to_csv(split_csv_path, index=False)
            print(f"Saved {split_name} split: {split_csv_path} ({len(df)} rows)")

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
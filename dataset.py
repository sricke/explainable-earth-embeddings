import logging
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader

from matplotlib.pyplot import Figure
from utils import set_seed

logger = logging.getLogger(__name__)


class GeoTextDataset(Dataset):

    def __init__(
        self,
        root: Path = 'data',
        split: str = 'train',
        subset_size: int = None,
        precomputed_text_embeddings: bool = True,
        precomputed_location_embeddings: bool = True,
        precomputed_dir: Path = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser()
        npy_root = Path(precomputed_dir).expanduser() if precomputed_dir else self.root

        loc_npy = npy_root / f"{split}_location_embeddings.npy"
        text_npy = npy_root / f"{split}_text_embeddings.npy"
        skip = {f"{t}_embedding" for t, p in [("location", loc_npy), ("text", text_npy)] if p.exists()}

        def load_parquet(path):
            cols = [c for c in pq.read_schema(path).names if c not in skip] if skip else None
            return pd.read_parquet(path, columns=cols)

        parquet_path = self.root / f"{split}.parquet"
        csv_path    = self.root / f"{split}.csv"
        data_parquet = self.root / "data.parquet"
        data_csv     = self.root / "data.csv"

        if parquet_path.exists():
            self.df = load_parquet(parquet_path)
        elif csv_path.exists():
            self.df = pd.read_csv(csv_path, index_col=False)
        elif data_parquet.exists():
            self._save_train_test_split(load_parquet(data_parquet), fmt="parquet")
            self.df = load_parquet(self.root / f"{split}.parquet")
        elif data_csv.exists():
            self._save_train_test_split(pd.read_csv(data_csv, index_col=False), fmt="csv")
            self.df = pd.read_csv(self.root / f"{split}.csv", index_col=False)
        else:
            raise FileNotFoundError(f"No data found in {self.root}")

        self.loc_embeddings  = np.load(loc_npy,  mmap_mode='r') if loc_npy.exists()  else None
        self.text_embeddings = np.load(text_npy, mmap_mode='r') if text_npy.exists() else None
        self.indices = np.arange(len(self.df))

        if subset_size is not None:
            self.subsample(subset_size)
            if self.loc_embeddings is not None or self.text_embeddings is not None:
                self._materialize_subset(npy_root, split)

        np.random.default_rng(seed).shuffle(self.indices)

        self.precomputed_location_embeddings = precomputed_location_embeddings and (
            self.loc_embeddings is not None or 'location_embedding' in self.df.columns)
        self.precomputed_text_embeddings = precomputed_text_embeddings and (
            self.text_embeddings is not None or 'text_embedding' in self.df.columns)

        if not self.precomputed_text_embeddings:
            assert 'text' in self.df.columns or 'caption' in self.df.columns
        if not self.precomputed_location_embeddings:
            assert all(c in self.df.columns for c in ['lat', 'lon'])

        self._need_df_row = (
            (self.precomputed_location_embeddings and self.loc_embeddings is None) or
            (not self.precomputed_location_embeddings) or
            (self.precomputed_text_embeddings and self.text_embeddings is None) or
            (not self.precomputed_text_embeddings)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index, return_dict=False):
        i = int(self.indices[index])
        row = self.df.iloc[i] if self._need_df_row else None

        if self.precomputed_location_embeddings:
            src = self.loc_embeddings[i] if self.loc_embeddings is not None else row['location_embedding']
            loc, loc_key = torch.tensor(src, dtype=torch.float32), 'location_embedding'
        else:
            loc, loc_key = torch.tensor([row['lat'], row['lon']], dtype=torch.float32), 'latlon'

        if self.precomputed_text_embeddings:
            src = self.text_embeddings[i] if self.text_embeddings is not None else row['text_embedding']
            txt, txt_key = torch.tensor(src, dtype=torch.float32), 'text_embedding'
        else:
            txt, txt_key = row['text' if 'text' in self.df.columns else 'caption'], 'text'

        if return_dict:
            return {loc_key: loc, txt_key: txt}
        return loc, txt

    def reshuffle(self, seed: int):
        np.random.default_rng(seed).shuffle(self.indices)

    def _materialize_subset(self, npy_root: Path, split: str):
        import hashlib
        cache_dir = Path(npy_root) / ".subset_cache"
        cache_dir.mkdir(exist_ok=True)

        # Sort indices so reads are monotonically increasing through the file,
        # enabling OS read-ahead instead of random seeks across a 100+ GB file.
        sort_order = np.argsort(self.indices)
        sorted_indices = self.indices[sort_order]
        h = hashlib.md5(self.indices.tobytes()).hexdigest()[:12]

        if self.loc_embeddings is not None:
            path = cache_dir / f"{split}_loc_{h}.npy"
            if not path.exists():
                logger.info(f"Caching subset location embeddings -> {path}")
                np.save(path, self.loc_embeddings[sorted_indices])
            self.loc_embeddings = np.load(path)

        if self.text_embeddings is not None:
            path = cache_dir / f"{split}_txt_{h}.npy"
            if not path.exists():
                logger.info(f"Caching subset text embeddings -> {path}")
                np.save(path, self.text_embeddings[sorted_indices])
            self.text_embeddings = np.load(path)

        self.df = self.df.iloc[sorted_indices].reset_index(drop=True)
        self.indices = np.arange(len(self.df))

    def subsample(self, n: int, seed: int = 42):
        self.indices = np.random.default_rng(seed).choice(len(self.df), size=n, replace=False)

    def _save_train_test_split(self, all_data, val_size=0.1, test_size=0.1, random_state=42, fmt="csv"):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(all_data, test_size=test_size, random_state=random_state)
        train, val  = train_test_split(train, test_size=val_size, random_state=42)
        for name, df in {'train': train, 'val': val, 'test': test}.items():
            path = self.root / f"{name}.{fmt}"
            df.to_parquet(path, index=False) if fmt == "parquet" else df.to_csv(path, index=False)
            print(f"Saved {name}: {path} ({len(df)} rows)")
        return train, val, test

    def plot(self, color=None, s=5) -> Figure:
        import matplotlib.pyplot as plt
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots(1, figsize=(6, 3))
        Basemap(projection='cyl', resolution='c', ax=ax).drawcoastlines()
        coords = self.df[['lon', 'lat']].iloc[self.indices].to_numpy()
        ax.scatter(coords[:, 0], coords[:, 1], c=color, s=s, alpha=0.7)
        ax.set_title("Plot of data points")
        return fig


def build_dataloaders(
    dataset_path, batch_size, num_workers, seed,
    precomputed_text_embeddings=True, precomputed_location_embeddings=True,
    train_subsample_size=None, val_subsample_size=None, precomputed_dir=None,
):
    train_ds = GeoTextDataset(root=dataset_path, split='train', subset_size=train_subsample_size,
        precomputed_text_embeddings=precomputed_text_embeddings,
        precomputed_location_embeddings=precomputed_location_embeddings, precomputed_dir=precomputed_dir, seed=seed)
    val_ds = GeoTextDataset(root=dataset_path, split='val', subset_size=val_subsample_size,
        precomputed_text_embeddings=precomputed_text_embeddings,
        precomputed_location_embeddings=precomputed_location_embeddings, precomputed_dir=precomputed_dir, seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    worker_init = lambda wid: set_seed(seed + wid)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init, generator=g,
                          pin_memory=True, persistent_workers=num_workers > 0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=worker_init,
                          pin_memory=True, persistent_workers=num_workers > 0)
    logger.info(f"Loaded from {dataset_path}: train={len(train_ds)}, val={len(val_ds)}")
    return train_dl, val_dl

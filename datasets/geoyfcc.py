from pathlib import Path
from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset

from .base import BaseTextDataset


class GeoYFCCTextDataset(BaseTextDataset, Dataset):
    """
    GeoYFCC text dataset.

    Expected source pickle columns:
        lon, lat, text

    Usage:
        - If you already have a CSV, pass `csv_path`.
        - If `csv_path` is None or does not exist, provide `pkl_path`
          (and optionally `output_dir`) and the CSV splits will be
          created from the pickle in a reproducible way.
    """

    def __init__(
        self,
        split: str = "train",
        csv_path: Path | str | None = None,
        pkl_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        split = split.lower()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        # If CSV is not provided or missing, fall back to pickle and create splits.
        if csv_path is None or not Path(csv_path).expanduser().exists():
            if pkl_path is None:
                pkl_path = Path("~/data/geoyfcc/geoyfcc_text_filtered_single_label.pkl")
            else:
                pkl_path = Path(pkl_path)

            if output_dir is None:
                output_dir = Path("~/data/geoyfcc/output")
            else:
                output_dir = Path(output_dir)

            train_csv, val_csv = self.create_splits_from_pickle(
                pkl_path=pkl_path,
                output_dir=output_dir,
                test_size=test_size,
                random_state=random_state,
            )

            csv_path = train_csv if split == "train" else val_csv

        csv_path = Path(csv_path).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found and could not be created: {csv_path}")

        df = pd.read_csv(csv_path)
        # Ensure we only keep lat, lon, text and that required columns exist.
        self.df = self.validate_dataframe(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "lat": row["lat"],
            "lon": row["lon"],
            "text": row["text"],
        }

    # -------- Dataset preparation utilities -------- #

    @staticmethod
    def load_pickle(pkl_path: Path) -> pd.DataFrame:
        """Load and normalize GeoYFCC pickle."""
        pkl_path = Path(pkl_path).expanduser()
        if not pkl_path.exists():
            raise FileNotFoundError(f"GeoYFCC pickle not found: {pkl_path}")

        df = pd.read_pickle(pkl_path)

        required_cols = {"lat", "lon", "text"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"GeoYFCC pickle is missing required columns: {missing}"
            )

        # Keep only the minimal columns needed downstream.
        return df[["lat", "lon", "text"]]

    @classmethod
    def create_splits_from_pickle(
        cls,
        pkl_path: Path,
        output_dir: Path,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[Path, Path]:
        """Create train/val CSV splits directly from GeoYFCC pickle."""
        df = cls.load_pickle(pkl_path)
        df = cls.validate_dataframe(df)

        return cls.create_csv_splits(
            df=df,
            output_dir=Path(output_dir).expanduser(),
            test_size=test_size,
            random_state=random_state,
        )


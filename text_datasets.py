from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class BaseTextDataset:
    """Base utilities for text datasets."""

    # We only keep lat, lon and text to be minimal and model-agnostic.
    REQUIRED_COLUMNS = {"lat", "lon", "text"}

    @staticmethod
    def ensure_output_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataframe contains required columns."""
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Always return just the required columns, dropping rows with NaNs.
        return df[list(cls.REQUIRED_COLUMNS)].dropna()

    @classmethod
    def create_csv_splits(
        cls,
        df: pd.DataFrame,
        output_dir: Path,
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[Path, Path]:
        """Create train/val CSV splits."""
        cls.ensure_output_dir(output_dir)

        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )

        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        return train_path, val_path


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
        # Dataset currently has no explicit language column, so we cannot
        # reliably filter by language. If needed, language-based filtering
        # should be done upstream when creating the pickle.

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


if __name__ == "__main__":
    # Example usage

    pkl_path = Path("~/data/geoyfcc/geoyfcc_text_filtered_single_label.pkl")
    output_dir = Path("~/data/geoyfcc_geoclip_satclip")

    train_csv, val_csv = GeoYFCCTextDataset.create_splits_from_pickle(
        pkl_path=pkl_path,
        output_dir=output_dir,
    )

    print("Train CSV:", train_csv)
    print("Val CSV:", val_csv)
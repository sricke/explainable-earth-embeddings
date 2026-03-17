from pathlib import Path
from typing import Tuple

import pandas as pd


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
        from sklearn.model_selection import train_test_split

        cls.ensure_output_dir(output_dir)

        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=True
        )

        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)

        return train_path, val_path


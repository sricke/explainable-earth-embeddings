from pathlib import Path
from typing import Tuple

import pandas as pd
from torch.utils.data import Dataset

from .base import BaseTextDataset


class WildSatTextDataset(BaseTextDataset, Dataset):
    """
    WildSat text dataset.

    Supports two modes:
      1) Load from an already prepared split CSV containing columns
         `lat`, `lon`, `text` (plus optional metadata).
      2) Build train/val CSV splits from the raw WildSat CSV
         (e.g., geolocated_text_dataset.csv) and then load the selected split.

    Typical raw WildSat columns:
        taxon_id, species_name, section_name, section_type, text, lat, lon
    """

    def __init__(
        self,
        split: str = "train",
        csv_path: Path | str | None = None,
        wildsat_csv_path: Path | str | None = None,
        output_dir: Path | str | None = None,
        section_types: tuple[str, ...] | list[str] | None = ("range", "habitat"),
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        split = split.lower()
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        print(
            f"[WildSatTextDataset.__init__] split={split} "
            f"csv_path={csv_path!r} wildsat_csv_path={wildsat_csv_path!r} "
            f"output_dir={output_dir!r} section_types={section_types} "
            f"test_size={test_size} random_state={random_state}"
        )

        # If split CSV is not provided or missing, create it from raw WildSat CSV.
        if csv_path is None or not Path(csv_path).expanduser().exists():
            if wildsat_csv_path is None:
                raise ValueError(
                    "Provide either `csv_path` or `wildsat_csv_path` to build splits."
                )

            raw_path = Path(wildsat_csv_path).expanduser()
            if output_dir is None:
                output_dir = raw_path.parent
            output_dir = Path(output_dir).expanduser()

            print(
                f"[WildSatTextDataset.__init__] creating splits from raw CSV: "
                f"{raw_path} → output_dir={output_dir}"
            )

            train_csv, val_csv = self.create_splits_from_wildsat_csv(
                wildsat_csv_path=raw_path,
                output_dir=output_dir,
                section_types=section_types,
                test_size=test_size,
                random_state=random_state,
            )
            csv_path = train_csv if split == "train" else val_csv

        csv_path = Path(csv_path).expanduser()
        if not csv_path.exists():
            raise FileNotFoundError(f"WildSat split CSV not found: {csv_path}")

        print(f"[WildSatTextDataset.__init__] loading split CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(
            f"[WildSatTextDataset.__init__] loaded {len(df):,} rows from {csv_path.name}"
        )
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

    @classmethod
    def load_wildsat_csv(
        cls,
        wildsat_csv_path: Path,
        section_types: tuple[str, ...] | list[str] | None = ("range", "habitat"),
    ) -> pd.DataFrame:
        """Load raw WildSat CSV and normalize to minimal columns."""
        wildsat_csv_path = Path(wildsat_csv_path).expanduser()
        if not wildsat_csv_path.exists():
            raise FileNotFoundError(f"WildSat CSV not found: {wildsat_csv_path}")

        print(f"[WildSatTextDataset.load_wildsat_csv] reading {wildsat_csv_path}")
        df = pd.read_csv(wildsat_csv_path)
        print(
            f"[WildSatTextDataset.load_wildsat_csv] raw rows={len(df):,}, "
            f"columns={list(df.columns)}"
        )
        missing = cls.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"WildSat CSV is missing required columns: {missing}")

        if section_types is not None and "section_type" in df.columns:
            normalized = {s.lower() for s in section_types}
            print(
                "[WildSatTextDataset.load_wildsat_csv] filtering by "
                f"section_types={normalized}"
            )
            print(
                "[WildSatTextDataset.load_wildsat_csv] section_type value_counts:\n"
                f"{df['section_type'].value_counts(dropna=False).to_string()}"
            )
            df = df[df["section_type"].astype(str).str.lower().isin(normalized)]
            print(
                f"[WildSatTextDataset.load_wildsat_csv] rows after section_type "
                f"filter={len(df):,}"
            )

        return df[["lat", "lon", "text"]]

    @classmethod
    def create_splits_from_wildsat_csv(
        cls,
        wildsat_csv_path: Path,
        output_dir: Path,
        section_types: tuple[str, ...] | list[str] | None = ("range", "habitat"),
        test_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[Path, Path]:
        """Create train/val CSV splits directly from raw WildSat CSV."""
        df = cls.load_wildsat_csv(
            wildsat_csv_path=wildsat_csv_path,
            section_types=section_types,
        )
        df = cls.validate_dataframe(df)
        if df.empty:
            raise ValueError(
                "No WildSat rows remain after filtering/validation. "
                "Check `section_types` or input data quality."
            )

        print(
            "[WildSatTextDataset.create_splits_from_wildsat_csv] "
            f"rows after validation={len(df):,}, output_dir={output_dir}"
        )

        return cls.create_csv_splits(
            df=df,
            output_dir=Path(output_dir).expanduser(),
            test_size=test_size,
            random_state=random_state,
        )


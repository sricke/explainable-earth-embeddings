import pandas as pd

from .base_dataset import BaseLocDataset

class YFCCDataset(BaseLocDataset):
    def __init__(
        self,
        split,
        csv_path,
        value_col: str = "value",
        lat_col: str = "lat",
        lon_col: str = "lon",
        log_transformed: bool = False,
    ):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split]

        df = df.rename(
            columns={
                lat_col: "lat",
                lon_col: "lon",
                value_col: "value",
            }
        )

        super().__init__(df, log_transformed)
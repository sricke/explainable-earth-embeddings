from sklearn.model_selection import train_test_split
import pandas as pd

from .base_dataset import BaseLocDataset

class MosaiksDataset(BaseLocDataset):
    def __init__(
        self,
        split,
        csv_path,
        value_col: str = "value",
        lat_col: str = "lat",
        lon_col: str = "lon",
        seed: int = 42,
        log_transformed: bool = False,
    ):
        df = pd.read_csv(csv_path)
        df = df.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "value"})

        train, test = train_test_split(df, test_size=0.1, random_state=seed)
        train, val = train_test_split(train, test_size=0.1, random_state=seed)

        df_split = {"train": train, "val": val, "test": test}[split]

        super().__init__(df_split, log_transformed)
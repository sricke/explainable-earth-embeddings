from .base_dataset import BaseLocDataset

class SustainBenchDataset(BaseLocDataset):
    def __init__(
        self,
        split,
        trainval_path,
        test_path,
        value_col: str = "value",
        lat_col: str = "lat",
        lon_col: str = "lon",
        seed: int = 42,
        log_transformed: bool = False,
    ):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        if split == "test":
            df = pd.read_csv(test_path)
        else:
            df = pd.read_csv(trainval_path)
            train, val = train_test_split(df, test_size=0.1, random_state=seed)
            df = train if split == "train" else val

        df = df.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "value"})
        super().__init__(df, log_transformed)
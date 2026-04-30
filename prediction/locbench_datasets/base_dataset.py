import torch

from torch.utils.data import Dataset


class BaseLocDataset(Dataset):
    def __init__(self, df, log_transformed: bool = False):
        self.df, self.log_transformed = df.reset_index(drop=True), log_transformed
        missing_cols = {"lat", "lon", "value"} - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        loc = torch.tensor([row.lat, row.lon], dtype=torch.float32)
        val = torch.tensor(row.value, dtype=torch.float32)
        return loc, (torch.log1p(val) if self.log_transformed else val)


def dataset_kwargs(dataset: str) -> dict:
    return {"seed": 42} if dataset.split(".", 1)[0] in {"mosaiks", "sustainbench"} else {}
import json
import pandas as pd

from .base_dataset import BaseLocDataset

class EbirdDataset(BaseLocDataset):
    def __init__(
        self,
        split,
        meta_path,
        value_col: str = "class_id",
        lat_col: str = "ebird_meta.lat",
        lon_col: str = "ebird_meta.lon",
        log_transformed: bool = False,
    ):
        data = json.loads(open(meta_path).read())
        df = pd.json_normalize(data[split])

        if lat_col not in df.columns: lat_col = "ebird_meta.lat"
        if lon_col not in df.columns: lon_col = "ebird_meta.lon"
        df = df.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "value"})

        super().__init__(df, log_transformed)
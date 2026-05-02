import json
import pandas as pd

from .base_dataset import BaseLocDataset

class FMoWDataset(BaseLocDataset):
    def __init__(
        self,
        split,
        data_path,
        annotation_path,
        value_col: str = "category_id",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        log_transformed: bool = False,
    ):
        data = json.loads(open(data_path).read())
        df = pd.json_normalize(data)

        anno = json.loads(open(annotation_path).read())
        ann_df = pd.DataFrame(anno["annotations"])[["image_id", "category_id"]]

        df = df.merge(ann_df, left_on="id", right_on="image_id", how="left")

        df = df.rename(columns={lat_col: "lat", lon_col: "lon", value_col: "value"})

        if "split" in df.columns:
            df = df[df["split"] == split]

        super().__init__(df, log_transformed)
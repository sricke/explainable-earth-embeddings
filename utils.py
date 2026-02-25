from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def train_split(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    output_train = file_path.parent / "train.csv"
    output_val = file_path.parent / "val.csv"

    df_train.to_csv(output_train, index=False)
    df_test.to_csv(output_val, index=False)
    return df_train, df_test

def get_location_model_output_dim(location_model):
    embedding = location_model(torch.randn(1, 2)) # dummy location
    output_dim = embedding.shape[1]
    return output_dim

if __name__ == "__main__":
    file_path = Path("../../data/s2-100k/wikipedia-dataset/dataset.csv")
    train_split(file_path)
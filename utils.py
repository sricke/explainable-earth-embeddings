import pandas as pd
from sklearn.model_selection import train_test_split
import torch
def train_split(folder_path, test_size=0.2, random_state=42):
    df = pd.read_csv(f"{folder_path}/index_with_descriptions.csv")
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train.to_csv(f"{folder_path}/train.csv", index=False)
    df_test.to_csv(f"{folder_path}/val.csv", index=False)
    return df_train, df_test

def get_location_model_output_dim(location_model):
    embedding = location_model(torch.randn(1, 2)) # dummy location
    output_dim = embedding.shape[1]
    return output_dim

if __name__ == "__main__":
    train_split("/home/ricke/satcam/explainable-earth-embeddings/s2-100k")
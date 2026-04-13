import argparse
import torch
from dataset import GeoTextDataset
from models.model import TextLocationModel
from loss import CLIPLoss, ConceptLoss
from train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--text_encoder', type=str, required=True)
    parser.add_argument('--location_encoder', type=str, required=True)
    parser.add_argument('--train_loss', type=str, required=True)
    parser.add_argument('--model_save_path', type=str)
    args = parser.parse_args()

    dataset = GeoTextDataset(root=args.dataset_path)

    model = TextLocationModel()
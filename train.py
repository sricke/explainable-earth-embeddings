import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def train(train_dataloader, model, criterion, optimizer, epoch, device, scheduler=None):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    for i ,(texts, locs) in bar:
        text_features, location_features = model(texts, locs)

        loss = criterion(text_features, location_features)
        optimizer.zero_grad()

        # Backpropagate
        loss.backward()
        optimizer.step()

        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, loss.item()))

    if scheduler is not None:
        scheduler.step()
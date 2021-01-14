import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset


def get_dataloader(train=True, pin_memory=True, batch_size=128, **kwargs):
    pin_memory = pin_memory and torch.cuda.is_available
    dataset = TimeseriesDataset(path, train=train)

    if train:
        dataloader_train = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, **kwargs)
        dataloader_valid = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, **kwargs)

        return dataloader_train, dataloader_valid

    else:
        dataloader_test = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, **kwargs)

        return dataloader_test


class TimeseriesDataset(Dataset):

    def __init__(self, x_path, y_path, train=True):
        print("***PREPARING DATASET***")
        self.x = pd.read_csv(x_path)
        self.y = pd.read_csv(y_path)
        self.train = train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.train:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange

from dataloading import get_dataloader


DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'


def evaluate(dataloader, model, criterion, epoch=0):
    loss_epoch_eval = 0

    with trange(len(dataloader), desc=f"Valid epoch {epoch+1}", leave=False) as t:
        for x, y in dataloader:
            x.to(DEVICE)
            y.to(DEVICE)

            y_pred = model(x)
            loss_batch = criterion(y, y_pred)
            loss_epoch_eval += loss_batch.item()

            t.set_postfix(loss=loss_batch.item())
            t.update()

    return loss_epoch_eval / len(dataloader), metrics


def train(dataloader_train, dataloader_valid, model, criterion, optimizer, *args):
    best_loss = np.inf
    early_stopping_count = 0

    for epoch in range(args.epochs):
        loss_epoch_train = 0

        with trange(len(dataloader_train), desc=f"Train epoch {epoch+1}", leave=False) as t:
            for x, y in dataloader_train:
                x.to(DEVICE)
                y.to(DEVICE)

                y_pred = model(x)
                loss_batch = criterion(y, y_pred)
                loss_epoch_train += loss_batch.item()

                optimizer.zero_grad()
                loss_batch.backwards()
                optimizer.step()

                t.set_postfix(loss=loss_batch.item())
                t.update()

        loss_epoch_train /= len(dataloader_train)
        loss_epoch_valid, metrics = evaluate(dataloader_valid, model, criterion, epoch=epoch)

        if loss_epoch_valid < best_loss:
            torch.save(model, os.path.join(args.save_path, "best_model.pt"))
            best_loss = loss_epoch_valid
        else:
            early_stopping_count += 1
        
        if early_stopping_count == 5:
            break


class Net(nn.Module):
    def __init__(self, input_size, *args):
        super().__init__()

        self.project = nn.Linear(input_size, args.embed_dim)
        self.gru = nn.GRU(
            input_size=args.embed_dim,
            hidden_size=args.gru_hidden_size,
            num_layers=args.gru_num_layers)
        self.output = nn.Linear(args.gru_hidden_size, args.output_size)

    def forward(self, x):
        x = self.project(x)
        x = self.gru(x)
        x = self.output(x)

        return x


if __name__ == "__main__":
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str)
    parser.add_argument('-f', '--fold', type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', 'batch-size', type=int, default=128)
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001)
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args(args)

    if not args.test:
        dataloader_train, dataloader_valid = get_dataloader()
        model = Net(input_size=dataloader_train.dataset.shape[-1], *args)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(params=model.params)

        train(dataloader_train, dataloader_valid, model, criterion, optimizer, *args)

    else:
        dataloader_test = get_dataloader(train=False)
        model = torch.load(args.model_path)
        criterion = nn.BCEWithLogitsLoss()

        evaluate(dataloader_test, model, criterion)

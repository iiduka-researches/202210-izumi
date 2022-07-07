import mailbox
import os
from typing import Tuple, Optional

import torch
from torch.nn import Embedding, Linear, LSTM, Module, BCEWithLogitsLoss, AlphaDropout
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchtext.data import LabelField, BucketIterator, Field
from torchtext.datasets import IMDB

from .base import BaseExperiment, ResultDict


class ExperimentIMDb(BaseExperiment):
    def __init__(self, dataset_name='IMDb', **kwargs):
        super(ExperimentIMDb, self).__init__(dataset_name=dataset_name, **kwargs)

        # DL the Dataset and split
        self.text = Field(sequential=True, fix_length=80, batch_first=True, lower=True)
        self.label = LabelField(sequential=False)
        self.train_data, self.test_data = IMDB.splits(root=self.data_dir, text_field=self.text, label_field=self.label)

        # build the vocabulary
        self.text.build_vocab(self.train_data, max_size=25000)
        self.label.build_vocab(self.train_data)
        self.vocab_size = len(self.text.vocab)

    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        pass

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        return Net(in_dim=self.vocab_size, **kwargs)

    def epoch_train(self, net: Module, optimizer: Optimizer,
                    train_loader: data.DataLoader, **kwargs) -> Tuple[Module, ResultDict]:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device, dtype=torch.long)
            labels = labels.to(self.device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=None)
            running_loss += loss.item()
            total += labels.size(0)
            predicted = torch.where(outputs <= .5, torch.zeros_like(outputs), torch.ones_like(outputs))
            correct += (predicted == labels).sum().item()
            i += 1
        return net, dict(train_loss=running_loss / i, train_accuracy=correct / total)

    def epoch_validate(self, net: Module, test_loader: data.DataLoader, **kwargs) -> ResultDict:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = BCEWithLogitsLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device, dtype=torch.long)
                labels = labels.to(self.device, dtype=torch.float)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = torch.where(outputs <= .5, torch.zeros_like(outputs), torch.ones_like(outputs))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1
        return dict(test_loss=running_loss / i, test_accuracy=correct / total)

    def prepare_loaders(self):
        return BucketIterator.splits((self.train_data, self.test_data), batch_size=self.batch_size, device=self.device,
                                     sort_key=lambda x: len(x.text), repeat=False)


class Net(Module):
    def __init__(self, in_dim: int, embedding_dim=50, hidden_size=50, num_layers=2) -> None:
        super().__init__()
        self.emb = Embedding(in_dim, embedding_dim, padding_idx=0)
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.drop = AlphaDropout(p=0.2)
        self.linear = Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        x = torch.cat([h[0], h[-1]], dim=1)
        x = self.drop(x)
        x = self.linear(x)
        return x.squeeze()

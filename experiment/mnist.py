from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.utils import data
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from experiment.base import BaseExperiment, Optimizer, ResultDict


class ExperimentMNIST(BaseExperiment):
    def __init__(self, **kwargs) -> None:
        super(ExperimentMNIST, self).__init__(dataset_name='MNIST', **kwargs)

    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        return MNIST(root=self.data_dir, train=train, download=True, transform=ToTensor(), **kwargs)

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name == 'Perceptron1':
            return Linear(n_hidden=1, **kwargs)
        else:
            return Linear(n_hidden=2, **kwargs)

    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: data.DataLoader,
                    **kwargs) -> Tuple[Module, ResultDict]:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = CrossEntropyLoss()
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step(closure=None)
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            i += 1

        return net, dict(train_loss=running_loss / i, train_accuracy=correct / total)

    def epoch_validate(self, net: Module, test_loader: data.DataLoader, **kwargs) -> ResultDict:
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = CrossEntropyLoss()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                i += 1
        return dict(test_loss=running_loss / i, test_accuracy=correct / total)


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
        )
        self.mlp = nn.Sequential(
            nn.Linear(1024, 200),
            nn.Dropout(0.25),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        out = self.conv_net(x)
        out = out.view(out.size(0), -1)
        out = self.mlp(out)
        return out


class Linear(nn.Module):
    def __init__(self,  in_dim=784, n_hidden=2, out_dim=10) -> None:
        super(Linear, self).__init__()
        if n_hidden == 1:
            self.linear = nn.Sequential(
                nn.Linear(in_dim, 100),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(100, out_dim),
            )
        elif n_hidden == 2:
            self.linear = nn.Sequential(
                nn.Linear(in_dim, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, out_dim),
            )
        else:
            raise ValueError(f'n_hidden should be 1 or 2, but n_hidden = {n_hidden}.')

    def forward(self, x: torch.Tensor):
        _, c, h, w = x.shape
        m = x.reshape(-1, c * h * w)
        return self.linear(m)

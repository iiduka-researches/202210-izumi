from typing import *

import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.transforms import ToTensor

from experiment.base import BaseExperiment, LossNaError, ResultDict
from model.densenet import densenet_bc
from model.resnet import resnet20, resnet32, resnet44, resnet56, resnet110


MODEL_DICT: Dict[str, Callable] = dict(
    ResNet20=resnet20,
    ResNet32=resnet32,
    ResNet44=resnet44,
    ResNet56=resnet56,
    ResNet110=resnet110,
    DenseNetBC24=densenet_bc,

    ResNet18=resnet18,
    ResNet34=resnet34,
    ResNet50=resnet50,
    ResNet101=resnet101,
    ResNet152=resnet152,
    DenseNet121=densenet121,
    DenseNet161=densenet161,
    DenseNet169=densenet169,
    DenseNet201=densenet201,
)


class ExperimentCIFAR10(BaseExperiment):
    def __init__(self, dataset_name='CIFAR10', **kwargs) -> None:
        super(ExperimentCIFAR10, self).__init__(dataset_name=dataset_name, **kwargs)

    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        return CIFAR10(root=self.data_dir, train=train, download=True, transform=ToTensor(), **kwargs)

    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        if model_name and model_name in MODEL_DICT:
            return MODEL_DICT[model_name](**kwargs)
        else:
            raise ValueError(f'Invalid model name: {model_name}')

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

            # debug
            if loss != loss:
                from utils.line.notify import notify
                notify(f'{i}: loss is NaN...')
                raise LossNaError('loss is NaN...')

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

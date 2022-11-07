from typing import *

import torch
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torchvision.datasets import CIFAR100
from model.resnet_pytorch100 import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from experiment.base import BaseExperiment, LossNaError, ResultDict
from model.resnetc100 import resnet20, resnet32, resnet44, resnet56, resnet110


MODEL_DICT: Dict[str, Callable] = dict(
    ResNet20=resnet20,
    ResNet32=resnet32,
    ResNet44=resnet44,
    ResNet56=resnet56,
    ResNet110=resnet110,

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


class ExperimentCIFAR100(BaseExperiment):
    def __init__(self, dataset_name='CIFAR100', **kwargs) -> None:
        super(ExperimentCIFAR100, self).__init__(dataset_name=dataset_name, **kwargs)

    def prepare_data(self, train: bool, **kwargs):
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5088964127604166, 0.48739301317401956, 0.44194221124387256), (0.2682515741720801, 0.2573637364478126, 0.2770957707973042))
        ])
        if train == 'True':
            trainset = CIFAR100(root=self.data_dir, train=train, download=True, transform=transform_train)
            return trainset
        else:
            testset = CIFAR100(root=self.data_dir, train=train, download=True, transform=transform_test)
            return testset

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
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
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
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                i += 1
        return dict(test_loss=running_loss / i, test_accuracy=correct / total)

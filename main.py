from typing import *
from torch.optim.lr_scheduler import *


from experiment.cifar10 import ExperimentCIFAR10
from experiment.cifar100 import ExperimentCIFAR100
from experiment.mnist import ExperimentMNIST
from experiment.imdb import ExperimentIMDb

from optimizer.adam import Adam
from optimizer.adam2 import Adam2
from optimizer.SCGAdam import SCGAdam
from optimizer.SCGAdam2 import SCGAdam2
from optimizer.sgd import SGD
from optimizer.sgd2 import SGD2
from optimizer.RMSProp import RMSprop
from optimizer.RMSProp2 import RMSprop2
from optimizer.adagrad import Adagrad
from optimizer.adagrad2 import Adagrad2
from optimizer.adamw import AdamW
from optimizer.adamw2 import AdamW2

import torch.optim as optim


Optimizer = Union[SGD, SGD2, RMSprop, RMSprop2, Adagrad, Adagrad2, AdamW, AdamW2, Adam, Adam2, SCGAdam, SCGAdam2,] 
OptimizerDict = Dict[str, Tuple[Any, Dict[str, Any]]]

def prepare_optimizers(lr: float, optimizer: str = None, **kwargs) -> OptimizerDict:
    optimizers = dict(
        Momentum_Cosinelr1e1wd5e4=(SGD, dict(lr=1e-1, momentum=0.9, **kwargs)),
        Adam_Cosinelr1e3=(Adam, dict(lr=1e-3, amsgrad=False, **kwargs)),
        AdamW_Cosinelr1e3=(AdamW, dict(lr=1e-3, weight=1e-2, amsgrad=False, **kwargs)),
        AMSGrad_Cosinelr1e3=(Adam, dict(lr=1e-3, amsgrad=True, **kwargs)),
        SGD_Cosinelr1e2=(SGD, dict(lr=1e-2, **kwargs)),
        RMSProp_Cosinelr1e2=(RMSprop, dict(lr=1e-2, alpha=0.99, **kwargs)),
        Adagrad_Cosinelr1e2=(Adagrad, dict(lr=1e-2, **kwargs)),
        SCGAdam_C1C3Cosinelr1e3=(SCGAdam, dict(lr=1e-3, amsgrad=False, cg1_type='C1', cg2_type='C3', **kwargs)),
        SCGAMSG_C1C3Cosinelr1e3=(SCGAdam, dict(lr=1e-3, amsgrad=True, cg1_type='C1', cg2_type='C3', **kwargs)),
        SCGAdam_C1C2Cosinelr1e3=(SCGAdam, dict(lr=1e-3, amsgrad=False, cg1_type='C1', cg2_type='C2', **kwargs)),
        SCGAMSG_C1C2Cosinelr1e3=(SCGAdam, dict(lr=1e-3, amsgrad=True, cg1_type='C1', cg2_type='C2', **kwargs)),

        SGDD0_Existing=(SGD2, dict(lr='D0', momentum='No', **kwargs)),
        MomentumD0_Existing=(SGD2, dict(lr='D0', momentum='D1', **kwargs)),
        RMSPropD0_Existing=(RMSprop2, dict(lr='D0', alpha='D1', **kwargs)),
        AdagradD0_Existing=(Adagrad2, dict(lr='D0', **kwargs)),
        AdamWD0_Existing=(AdamW2, dict(lr='D0', beta='D1', weight=1e-2, amsgrad=False, **kwargs)),
        AdamD0_Existing=(Adam2, dict(lr='D0', beta='D1', amsgrad=False, **kwargs)),
        AMSGradD0_Existing=(Adam2, dict(lr='D0', beta='D1', amsgrad=True, **kwargs)),
        SCGAdam_D1=(SCGAdam2, dict(lr='D0', beta='D1', amsgrad=False, cg1_type='D1', cg2_type='D1', **kwargs)),
        SCGAMSG_D1=(SCGAdam2, dict(lr='D0', beta='D1', amsgrad=True, cg1_type='D1', cg2_type='D1', **kwargs)),

    )
    if optimizer:
        return {optimizer: optimizers[optimizer]}
    else:
        return optimizers

def lr_warm_up(epoch: int, lr: float, t: int = 5, c: float = 1e-2):
    if epoch <= t:
        return ((1 - c) * epoch / t + c) * lr

    else:
        return lr


def lr_divide(epoch: int, max_epoch: int, lr: float):
    p = epoch / max_epoch
    if p < .5:
        return lr
    elif p < .75:
        return lr * 1e-1
    else:
        return lr * 1e-2


def lr_warm_up_divide(epoch: int, max_epoch: int, lr: float, t: int = 5, c: float = 1e-2):
    if epoch <= t:
        return lr_warm_up(epoch, lr, t, c)
    else:
        return lr_divide(epoch, max_epoch, lr)


def imdb(lr=1e-3, max_epoch=100, weight_decay=.0, batch_size=128, use_scheduler=False, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    e = ExperimentIMDb(max_epoch=max_epoch, batch_size=batch_size, **kwargs)
    e.execute(optimizers)



def mnist(lr=1e-3, max_epoch=100, weight_decay=.0, batch_size=128, model_name='Perceptron2', use_scheduler=False, **kwargs) -> None:
    optimizers = prepare_optimizers(lr=lr)
    scheduler = ReduceLROnPlateau if use_scheduler else None
    e = ExperimentMNIST(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name, scheduler=scheduler,
                        **kwargs)
    e.execute(optimizers)


def cifar10(max_epoch=200, lr=1e-3, weight_decay=0, batch_size=128, model_name='ResNet18', num_workers=1,
            optimizer=None, use_scheduler=True, **kwargs) -> None:
    scheduler = CosineAnnealingLR if use_scheduler else None
    optimizers = prepare_optimizers(lr=lr, optimizer=optimizer, weight_decay=weight_decay)
    e = ExperimentCIFAR10(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name,
                          kw_loader=dict(num_workers=num_workers), scheduler=scheduler,
                          **kwargs)
    e(optimizers)

def cifar100(max_epoch=200, lr=1e-3, weight_decay=0, batch_size=128, model_name='ResNet18', num_workers=1,
            optimizer=None, use_scheduler=True, **kwargs) -> None:
    scheduler = CosineAnnealingLR if use_scheduler else None
    optimizers = prepare_optimizers(lr=lr, optimizer=optimizer, weight_decay=weight_decay)
    e = ExperimentCIFAR100(max_epoch=max_epoch, batch_size=batch_size, model_name=model_name,
                          kw_loader=dict(num_workers=num_workers), scheduler=scheduler,
                          **kwargs)
    e(optimizers)


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('-e', '--experiment')
    p.add_argument('-m', '--model_name')
    p.add_argument('-d', '--data_dir', default='dataset/data')
    p.add_argument('-me', '--max_epoch', type=int)
    p.add_argument('-bs', '--batch_size', type=int)
    p.add_argument('--lr', type=float)
    p.add_argument('--device')
    p.add_argument('-nw', '--num_workers', type=int)
    p.add_argument('-us', '--use_scheduler', action='store_true')
    p.add_argument('-o', '--optimizer', default=None)
    p.add_argument('-wd', '--weight_decay', default=0, type=float)
    args = p.parse_args()

    experiment = args.experiment
    kw = {k: v for k, v in dict(**args.__dict__).items() if k != 'experiment' and v is not None}
    print(kw)
    d: Dict[str, Callable] = dict(
        IMDB=imdb,
        CIFAR10=cifar10,
        CIFAR100=cifar100,
        MNIST=mnist,
    )
    d[experiment](**kw)

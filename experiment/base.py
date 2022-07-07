from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
import json
import os
import random
from time import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from pandas import DataFrame
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils import data
from tqdm import tqdm

from utils.line.notify import notify, notify_error

ParamDict = Dict[str, Any]
OptimDict = Dict[str, Tuple[Any, ParamDict]]
ResultDict = Dict[str, Any]
Result = Dict[str, Sequence[Any]]

SEP = '_'


class LossNaError(Exception):
    pass


class BaseExperiment(ABC, metaclass=ABCMeta):
    def __init__(self, batch_size: int, max_epoch: int, dataset_name: str, kw_dataset=None, kw_loader=None,
                 model_name='model', kw_model=None, kw_optimizer=None, scheduler=None, kw_scheduler=None,
                 data_dir='./dataset/data/', result_dir='./result', device=None) -> None:
        r"""Base class for all experiments.

        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.data_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(self.data_dir, exist_ok=True)

        _kw_dataset = kw_dataset if kw_dataset else dict()
        self.train_data = self.prepare_data(train=True, **_kw_dataset)
        self.test_data = self.prepare_data(train=False, **_kw_dataset)
        self.kw_loader = kw_loader if kw_loader else dict()

        self.model_name = model_name
        self.kw_model = kw_model if kw_model else dict()
        self.kw_optimizer = kw_optimizer if kw_optimizer else dict()
        self.scheduler = scheduler
        self.kw_scheduler = kw_scheduler if kw_scheduler else dict()

        self.device = device if device else select_device()

        self.result_dir = os.path.join(result_dir, dataset_name, model_name)
        os.makedirs(self.result_dir, exist_ok=True)

    def __call__(self, *args, **kwargs) -> None:
        self.execute(*args, **kwargs)

    @abstractmethod
    def prepare_data(self, train: bool, **kwargs) -> data.Dataset:
        raise NotImplementedError

    @abstractmethod
    def prepare_model(self, model_name: Optional[str], **kwargs) -> Module:
        raise NotImplementedError

    @abstractmethod
    def epoch_train(self, net: Module, optimizer: Optimizer, train_loader: data.DataLoader,
                    **kwargs) -> Tuple[Module, ResultDict]:
        raise NotImplementedError

    @abstractmethod
    def epoch_validate(self, net: Module, test_loader: data.DataLoader, **kwargs) -> ResultDict:
        raise NotImplementedError

    def prepare_loaders(self):
        train_loader = data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                       worker_init_fn=worker_init_fn, **self.kw_loader)
        test_loader = data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False,
                                      worker_init_fn=worker_init_fn, **self.kw_loader)
        return train_loader, test_loader

    def train(self, net: Module, optimizer: Optimizer, train_loader: data.DataLoader,
              test_loader: data.DataLoader) -> Tuple[Module, Result]:
        if self.scheduler:
            scheduler = self.scheduler(optimizer, **self.kw_scheduler)
        else:
            scheduler = None

        results = []
        for epoch in tqdm(range(self.max_epoch)):
            start = time()
            try:
                net, train_result = self.epoch_train(net, optimizer=optimizer, train_loader=train_loader)
            except LossNaError as e:
                print(e)
                break
            validate_result = self.epoch_validate(net, test_loader=test_loader)
            result = arrange_result_as_dict(t=time() - start, train=train_result, validate=validate_result)
            results.append(result)
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_result['train_loss'])
                else:
                    scheduler.step()
                notify(f'{scheduler.__dict__}')
            if epoch % 10 == 0:
                notify(f'{epoch}{result}')

        return net, concat_dicts(results)

    @notify_error
    def execute(self, optimizers: OptimDict, seed=0, checkpoint_dict: Dict[str, str] = None) -> None:
        train_loader, test_loader = self.prepare_loaders()
        period = len(train_loader)
        print(period)  # debug
        with open(os.path.join(self.result_dir, 'args.json'), 'w') as fp:
            json.dump({k: str(v) for k, v in dict(**self.__dict__).items()}, fp)

        for name, (optimizer_cls, kw_optimizer) in optimizers.items():
            path = os.path.join(self.result_dir, result_format(name))
            checkpoint_dir = os.path.join(self.result_dir, f'checkpoint/{name}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            if exist_result(name, self.result_dir):
                notify(f'{name} already exists.')
                continue
            else:
                fix_seed(seed)

                if 'SCG' in name:
                    kw_optimizer['period'] = period

                net = self.prepare_model(self.model_name, **self.kw_model)
                net.to(self.device)

                optimizer = optimizer_cls(net.parameters(), **kw_optimizer, **self.kw_optimizer)
                kw_optimizer_default = dict(**optimizer.__dict__)['defaults']
                notify(f'{name} {kw_optimizer_default}')

            # load model in case that specify checkpoint
            if checkpoint_dict and checkpoint_dict.get(name):
                model_path = os.path.join(checkpoint_dir, f'model_{checkpoint_dict[name]}')
                net = net.load_state_dict(torch.load(model_path))
                optimizer_path = os.path.join(checkpoint_dir, f'optimizer_{checkpoint_dict[name]}')
                optimizer.load_state_dict(torch.load(optimizer_path))

            net, result = self.train(net=net, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader)
            result_to_csv(result, name=name, kw_optimizer=kw_optimizer_default, path=path)

            # torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'model_{self.max_epoch}'))
            # torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f'optimizer_{self.max_epoch}'))

            # Expect error between Stochastic CG and Deterministic CG
            """
            if type(optimizer) in (CoBA, CoBA2):
                s = '\n'.join([str(e) for e in optimizer.scg_expect_errors])
                with open(os.path.join(self.result_dir, f'scg_expect_errors_{name}.csv'), 'w') as f:
                    f.write(s)
            """
            notify(f'finish: {name}.')


def select_device() -> str:
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Using {device} ...')
    return device


def arrange_result_as_dict(t: float, train: Dict[str, float], validate: Dict[str, float]) -> Dict[str, float]:
    train = {k if 'train' in k else f'train_{k}': v for k, v in train.items()}
    validate = {k if 'test' in k else f'test_{k}': v for k, v in validate.items()}
    return dict(time=t, **train, **validate)


def concat_dicts(results: Sequence[ResultDict]) -> Result:
    keys = results[0].keys()
    return {k: [r[k] for r in results] for k in keys}


def fix_seed(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    random.seed(worker_id)


def result_format(name: str, sep=SEP, extension='csv') -> str:
    ts = datetime.now().strftime('%y%m%d%H%M%S')
    return f'{name}{sep}{ts}.{extension}'


def exist_result(name: str, result_dir: str, sep=SEP) -> bool:
    for p in os.listdir(result_dir):
        if SEP.join(os.path.basename(p).split(sep)[:-1]) == name:
            return True
    return False


def result_to_csv(r: Result, name: str, kw_optimizer: ParamDict, path: str) -> None:
    df = DataFrame(r)
    df['optimizer'] = name
    df['optimizer_parameters'] = str(kw_optimizer)
    df['epoch'] = np.arange(1, df.shape[0] + 1)
    df.set_index(['optimizer', 'optimizer_parameters', 'epoch'], drop=True, inplace=True)
    df.to_csv(path, encoding='utf-8')

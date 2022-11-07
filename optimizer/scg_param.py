from typing import Any, Callable, Dict
import math


def get_scg1_param_fn(cg_type: str) -> Callable:
    return _scg1_param_dict[cg_type]

def get_scg2_param_fn(cg_type: str) -> Callable:
    return _scg2_param_dict[cg_type]

def get_scg3_param_fn(cg_type: str) -> Callable:
    return _scg3_param_dict[cg_type]

def get_scg4_param_fn(cg_type: str) -> Callable:
    return _scg4_param_dict[cg_type]



def cg_param_d0(n: int) -> float:
    return 1 / math.sqrt(n)

def cg_param_d1(n: int) -> float:
    return .5 ** n

def cg_param_d2(n: int) -> float:
    return 1 / n


_scg1_param_dict = dict(
    C1=lambda n: 1e-1,
    C2=lambda n: 1e-2,
    C3=lambda n: 1e-3,
    C4=lambda n: 1e-4,
    C5=lambda n: 1e-5,
    C6=lambda n: 1e-6,
    _1C=lambda n: 1.0,
    _2C=lambda n: 2.0,
    _10C=lambda n: 1e+1,
    _02C=lambda n: 2e-1,
    _9e2=lambda n: 9e-2,
    _5e2=lambda n: 5e-2,
    _5e3=lambda n: 5e-3,
)


_scg2_param_dict = dict(
    C1=lambda n: 1e-1,
    C2=lambda n: 1e-2,
    C3=lambda n: 1e-3,
    C4=lambda n: 1e-4,
    C5=lambda n: 1e-5,
    C6=lambda n: 1e-6,
    _1C=lambda n: 1.0,
    _2C=lambda n: 2.0,
    _10C=lambda n: 1e+1,
    _05C=lambda n: 5e-1,
    _5e2=lambda n: 5e-2,
    _5e3=lambda n: 5e-3,
)

_scg3_param_dict = dict(
    D0=cg_param_d0,
    D1=cg_param_d1,
    D2=cg_param_d2,
    No=lambda n: 0.,
)

_scg4_param_dict = dict(
    D0=cg_param_d0,
    D1=cg_param_d1,
    D2=cg_param_d2,
)
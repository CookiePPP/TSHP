from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.init import calculate_gain


def get_afunc(name: str, return_none=False):
    if return_none and name is None:
        return None
    
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name in ['leakyrelu', 'leaky_relu', 'lrelu']:
        return nn.LeakyReLU(0.1)
    if name == 'selu' or name == 'swish':
        return nn.SELU()
    if name == 'elu':
        return nn.ELU()
    if name == 'celu':
        return nn.CELU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'softplus':
        return nn.Softplus()
    if name == 'tanh':
        return torch.tanh
    if name == 'sigmoid':
        return torch.sigmoid
    raise NotImplementedError(f'can\'t find an act_func named "{name}"')

def get_afunc_gain(name: str) -> Tuple[str, float]:
    name = name.lower()
    if name == 'relu':
        return 'relu', 0.0
    if name in ['leakyrelu', 'leaky_relu', 'lrelu']:
        return 'leaky_relu', 0.0
    if name == 'selu' or name == 'swish':
        return 'selu', 0.0
    if name == 'elu':
        return 'relu', 0.0
    if name == 'celu':
        return 'relu', 0.0
    if name == 'gelu':
        return 'relu', 0.0
    if name == 'softplus':
        return 'relu', 0.0
    if name == 'tanh':
        return 'tanh', 0.0
    if name == 'sigmoid':
        return 'sigmoid', 0.0
    
    return 'linear', 0.0

from math import sqrt
from numpy import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from TSHP.utils.modules.core import ConvNorm, ResBlock, ConvTranspose, nnModule
from torch import Tensor
from typing import List, Tuple, Optional

class UNet(nnModule):
    """
    Takes input, cond and mask. Will downsample by downsample_factor, n_stages number of times, using (n_blocks*n_layers) convs.
    
    The downsampling allows much larger inputs to be summarised into high level features while not using absurd amounts of VRAM/compute.
    
    n_stages = number of times the input is downsampled
    n_blocks = how many blocks the input is ran through at each downsample
    n_layers = how many layers in each block.
        Residual connections are done over the input/outputs of the blocks so n_layers should never go over 5.
    downsample_factor = how much to downsample on each stage.
        2 is recommended
    hidden_dim = the initial hidden dim of the lowest layer with the largest length input
    channel_up_factor = how much to increase the channel dim on every downsample
        2 is recommended
    """
    def __init__(self,
                 n_stages:int, n_blocks:int, n_layers:int, kernel_size:int,
                 input_dim:int, hidden_dim:int, output_dim:int, cond_dim:int=0,
                 downsample_factor:int=2, nn_downsample=False, nn_upsample=False,
                 channel_up_factor:int=2, min_hidden_dim=1, max_hidden_dim=65536, conv_params=None,
        ):
        super().__init__()
        if conv_params is None:
            conv_params = {'rezero': True}
        if conv_params.get('act_func', None) is None:
            conv_params['act_func'] = 'leaky_relu'
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cond_dim   = cond_dim
        self.n_stages   = n_stages
        self.downsample_factor = downsample_factor
        if type(self.downsample_factor) in [list, tuple]:
            assert self.n_stages == len(self.downsample_factor), f'len of downsample_factor is {len(self.downsample_factor)}, expected {self.n_stages}'
        
        self.channel_up_factor = channel_up_factor
        
        # get hidden layers
        self.dn_stages = nn.ModuleList()
        self.up_stages = nn.ModuleList()
        self.dn_layers = nn.ModuleList() if nn_downsample else None
        self.up_layers = nn.ModuleList() if nn_upsample   else None
        in_dim = input_dim
        for i in range(n_stages):
            out_dim = min(max(hidden_dim*(channel_up_factor**i), min_hidden_dim), max_hidden_dim)
            first_stage = bool(i==0)
            last_stage  = bool(i+1==n_stages)
            self.dn_stages.append(
                ResBlock(in_dim, out_dim, out_dim,
                         n_blocks, n_layers, kernel_size=kernel_size, cond_dim=cond_dim, **conv_params)
            )
            self.up_stages.append(
                ResBlock(out_dim, output_dim if first_stage else in_dim, out_dim,
                         n_blocks, n_layers, kernel_size=kernel_size, cond_dim=cond_dim, **conv_params)
            )
            if self.dn_layers is not None and not last_stage:
                self.dn_layers.append(
                    ConvNorm(out_dim, out_dim,
                             kernel_size=self.get_dn_factor(i),
                             stride     =self.get_dn_factor(i), padding=0,)
                )
            if self.up_layers is not None and not last_stage:
                self.up_layers.append(
                    ConvTranspose(out_dim, out_dim,
                                  kernel_size=self.get_dn_factor(i),
                                  stride     =self.get_dn_factor(i), padding=0,)
                )
            in_dim = out_dim
    
    def get_dn_factor(self, i):
        return self.downsample_factor[i] if type(self.downsample_factor) in [tuple, list, torch.Tensor] else self.downsample_factor
    
    def dnsample(self, x, i):
        if self.dn_layers is None:
            x = F.avg_pool1d(
                x.transpose(1, 2),
                kernel_size=self.get_dn_factor(i),
                ceil_mode=True,
            ).transpose(1, 2)
        else:
            x = self.dn_layers[i](x)
        return x
    
    def upsample(self, x, i):
        if self.up_layers is None:
            x = F.interpolate(
                x.transpose(1, 2),
                scale_factor=self.get_dn_factor(i),
            ).transpose(1, 2)
        else:
            x = self.up_layers[i](x)
        return x
    
    def pad_to_factor(self, x):
        """Pad input to factor of (n_stages*downsample_factor)"""
        x_T = x.shape[-2]
        cumfactor = prod([self.get_dn_factor(i) for i in range(self.n_stages)])
        pad_T = (-x_T)%cumfactor
        if pad_T:
            return F.pad(x.transpose(1, 2), (0, pad_T)).transpose(1, 2)
        else:
            return x
    
    def downsample_to_list(self, x):
        x_list = [x,]
        for i in range(self.n_stages):
            x = F.avg_pool1d(x.transpose(1, 2), self.get_dn_factor(i), ceil_mode=True).transpose(1, 2)
            x_list.append(x)
        return x_list
    
    def get_mask_list(self, mask, type='ceil'):
        masks = self.downsample_to_list(mask.float())
        if   type=='floor':
            masks = [mask.floor().bool() for mask in masks]
        elif type=='round':
            masks = [mask.round().bool() for mask in masks]
        elif type== 'ceil':
            masks = [mask. ceil().bool() for mask in masks]
        else:
            raise NotImplementedError
        return masks
    
    def forward(self, x, cond=None, mask=None):# [B, T, D], [B, T, D], [B, T, 1]
        in_T = x.shape[1]
        x = self.pad_to_factor(x)
        if self.cond_dim:
            cond = self.pad_to_factor(cond)
            conds = self.downsample_to_list(cond)
        if mask is not None:
            mask = self.pad_to_factor(mask)
            masks = self.get_mask_list(mask)
        
        res_list = []
        for i, dn_stage in enumerate(self.dn_stages):
            last_stage = bool(i+1==self.n_stages)
            x = dn_stage(x,
                         conds[i] if cond is not None else None,
                         masks[i] if mask is not None else None,)
            res_list.append(x)
            if not last_stage:
                x = self.dnsample(x, i)
        
        skip_list = []
        for i, up_stage in reversed(list(enumerate(self.up_stages))):
            last_stage = bool(i+1==self.n_stages)
            if not last_stage:
                x = self.upsample(x, i)
            x = x + res_list[i]
            x = up_stage(x,
                         conds[i] if cond is not None else None,
                         masks[i] if mask is not None else None,)
            skip_list.append(x)
        
        if x.shape[1] != in_T:
            x = x[:, :in_T]# -> [B, :T, ...]
        assert x.shape[2] == self.output_dim, f'output dim is wrong shape! Got {x.shape[2]}, expected {self.output_dim}.\nThis is a code issue, please report to developer.'
        return x, skip_list# [[B, T//(downsample_factor**i), D], ]*n_stages


class DNNet(nnModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x, skip_list


class UPNet(nnModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x, skip_list
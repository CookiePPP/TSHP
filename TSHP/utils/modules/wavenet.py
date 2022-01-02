import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple, Optional
from TSHP.utils.modules.core import nnModule, ConvNorm
from TSHP.utils.modules.utils import get_mask1d

# "Gated Convolutional Neural Networks for Domain Adaptation"
#  https://arxiv.org/pdf/1905.06906.pdf
#@torch.jit.script
def GTU(input_a, input_b: Optional[Tensor], n_channels: int):
    """Gated Tanh Unit (GTU)"""
    if input_b is not None:
        in_act = input_a+input_b
    else:
        in_act = input_a
    t_act = torch.tanh(in_act[:, :, :n_channels])
    s_act = torch.sigmoid(in_act[:, :, n_channels:])
    acts = t_act * s_act
    return acts

#@torch.jit.script
def GTRU(input_a, input_b: Optional[Tensor], n_channels: int):# saves significant VRAM and runs faster, unstable for first 150K~ iters. (test a difference layer initialization?)
    """Gated[?] Tanh ReLU Unit (GTRU)"""
    if input_b is not None:
        in_act = input_a+input_b
    else:
        in_act = input_a
    t_act = torch.tanh(in_act[:, :, :n_channels])
    r_act = torch.nn.functional.relu(in_act[:, :, n_channels:], inplace=True)
    acts = t_act * r_act
    return acts

#@torch.jit.script
def GLU(input_a, input_b: Optional[Tensor], n_channels: int):
    """Gated Linear Unit (GLU)"""
    if input_b is not None:
        in_act = input_a+input_b
    else:
        in_act = input_a
    l_act = in_act[:, :, :n_channels]
    s_act = torch.sigmoid(in_act[:, :, n_channels:])
    acts = l_act * s_act
    return acts

def get_unit(name):
    dict = {'GLU': GLU, 'GTRU': GTRU, 'GTU': GTU,}
    return dict[name]

class DilatedWN(nnModule):
    def __init__(self, in_channels, out_channels, hidden_channels, cond_channels, n_blocks, n_layers,
                kernel_size=1, dropout=0.0, pre_kernel_size=1, res_kernel_size=1, dilation_base=2, separable=False, weight_norm=False,
                LSUV_init=False, partial_padding=False, rezero=False, res=True, causal=False, pad_right=False, gated_unit='GTU', post_act_func=None):
        super(DilatedWN, self).__init__()
        self.n_blocks = n_blocks# how many dilated cycles
        self.n_layers = n_layers# how long each dilated cycle should be. 2 = [1, 2], 4 = [1, 2, 4, 8], 6 = [1, 2, 4, 8, 16, 32], ... etc etc
        self.hidden_channels = hidden_channels
        self.cond_channels   = cond_channels
        self.dropout = dropout
        self.res = res# use residual connections
        if res_kernel_size == 'wn':
            res_kernel_size = kernel_size
        conv_params = {'partial_padding': partial_padding, 'LSUV_init': LSUV_init,}
        self.causal = causal
        self.pad_right = pad_right
        
        self.pre = ConvNorm(in_channels, hidden_channels, kernel_size=pre_kernel_size,
                            causal=causal, pad_right=pad_right, **conv_params)
        
        self.conacts = []; self.preacts = []; self.posacts = []
        for block_idx in range(self.n_blocks):
            is_first_block = bool(block_idx==0)
            for layer_idx in range(self.n_layers):
                is_last_layer = (block_idx+1==self.n_blocks) and (layer_idx+1==self.n_layers)
                dilation = int(round(dilation_base**layer_idx))
                self.preacts.append(
                    ConvNorm(hidden_channels, 2*hidden_channels, kernel_size=kernel_size, dilation=dilation, separable=separable, weight_norm=weight_norm,
                             causal=causal, pad_right=pad_right, **conv_params)
                )
                if self.cond_channels:
                    self.conacts.append(
                        ConvNorm(  cond_channels, 2*hidden_channels, kernel_size=1, **conv_params)
                    )
                self.posacts.append(
                    ConvNorm(hidden_channels, 2*hidden_channels if not is_last_layer else hidden_channels, kernel_size=res_kernel_size, separable=separable, weight_norm=weight_norm,
                             causal=causal, pad_right=pad_right, **conv_params)
                )
        self.preacts = nn.ModuleList(self.preacts)
        self.conacts = nn.ModuleList(self.conacts)
        self.posacts = nn.ModuleList(self.posacts)
        
        self.rezero = None
        if rezero:
            self.rezero = nn.Parameter(torch.ones(self.n_blocks*self.n_layers)*1e-2)
        
        self.gated_unit = get_unit(gated_unit or 'GTU')
        
        self.post1 = ConvNorm(hidden_channels, hidden_channels, kernel_size=pre_kernel_size, act_func=F.relu,
                              causal=causal, pad_right=pad_right, **conv_params)
        self.post1.conv.weight.data /= (self.n_blocks*self.n_layers)**0.5
        
        post_conv_params = conv_params.copy()
        if post_act_func is not None:
            post_conv_params['act_func'] = post_act_func
        self.post2 = ConvNorm(hidden_channels,    out_channels, kernel_size=pre_kernel_size,
                              causal=causal, pad_right=pad_right, **post_conv_params)
    
    def forward_cached_cond(self, x, c, lengths):
        self.cond_cached = c
        return self.forward(x, c, lengths)
    
    def maybe_shift(self, x, pad_right=False):
        if self.causal:
            x = x.transpose(1, 2)# [B, T, C] -> [B, C, T]
            if pad_right:
                x = F.pad(x, (-1, 1))# shift to the left
            else:
                x = F.pad(x, (1, -1))# shift to the right
            x = x.transpose(1, 2)# [B, C, T] -> [B, T, C]
        return x
    
    def forward(self, x, cond, lengths):# [B, T, iC], [B, T, cC], [B]
        if lengths is None:
            mask = torch.ones(x.shape[0], x.shape[1], 1)
        else:
            mask = ~get_mask1d(lengths)# [B, T, 1]
        
        x = self.maybe_shift(x, self.pad_right)
        x = self.pre(x.masked_fill(mask, 0.0)).masked_fill(mask, 0.0)# [B, T, iC] -> [B, T, hC]
        if self.cond_channels:
            c_len = cond.shape[1]
            if c_len != 1:
                cond = cond.masked_fill(mask, 0.0)
        
        if self.rezero is not None:
            rezero = self.rezero.unbind(0)# [B] -> list([], [], ...) - parameter tensor to list of vector parameters
        for block_idx in range(self.n_blocks):
            for layer_idx in range(self.n_layers):
                is_last_layer = (block_idx+1==self.n_blocks) and (layer_idx+1==self.n_layers)
                is_first_layer = block_idx+layer_idx==0
                idx = (block_idx*self.n_layers)+layer_idx
                
                x_res = x
                x  = self.preacts[idx](x).masked_fill_(mask, 0.0)# [B, T, hC] -> [B, T, 2*hC]
                cx = None
                if self.cond_channels:
                    cx = self.conacts[idx](cond)# [B, T, cC] -> [B, T, 2*hC]
                    if c_len != 1:
                        cx = cx.masked_fill_(mask, 0.0)
                x = self.gated_unit(x, cx, self.hidden_channels)# -> [B, T, hC]
                x = F.dropout(x, self.dropout, training=self.training)# dropout AFTER act_func
                
                x = self.posacts[idx](x).masked_fill_(mask, 0.0)# [B, T, cC] -> [B, T, 2*hC]
                if is_last_layer:
                    xskip = x# -> [B, T, hC]
                else:
                    x, xskip = x.chunk(2, dim=-1)# -> [B, T, hC], [B, T, hC]
                    if self.res:
                        if self.rezero is not None:
                            x = (x*rezero[idx]) + x_res
                        else:
                            x = (x_res + x)/(2**0.5)# so x_res + x doesn't increase the magnitudes of the latents
                skip = xskip if is_first_layer else xskip + skip
        
        x = self.post2(self.post1(skip).masked_fill(mask, 0.0)).masked_fill(mask, 0.0)# -> [B, T, oC]
        return x# [B, T, oC]
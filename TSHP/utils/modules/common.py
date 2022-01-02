import math
import warnings
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TSHP.utils.misc_utils import zip_equal
from TSHP.utils.modules.RevGrad import ScaleGrad
from TSHP.utils.modules.local_attention import SinusoidalEmbeddings, apply_rotary_pos_emb
from TSHP.utils.modules.loss_func.common import kld_loss
from TSHP.utils.modules.norms import BatchNorm1d
from torch import Tensor

from TSHP.utils.modules.activation_funcs import get_afunc
from TSHP.utils.modules.core import nnModule, reparameterize
from TSHP.utils.modules.utils import Fpad, get_mask, get_mask1d, maybe_cat


def dictify(**kwargs):# convert (param=value) to dict
    return kwargs

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


def get_weight_gain_from_act_func(act_func):
    if act_func is None:
        return 1.0
    elif type(act_func) is str: # name of act_func
        act_func_lower: str = act_func.lower()
        if act_func_lower == 'none':
            return 1.0
        elif act_func_lower in ['relu', 'elu', 'celu', 'gelu', 'softplus']:
            return math.sqrt(2.0)
        elif act_func_lower in ['leaky_relu', 'leakyrelu', 'lrelu']:
            negative_slope = 0.1
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        elif act_func_lower == 'tanh':
            return 5.0 / 3
        elif act_func_lower in ['selu', 'swish']:
            return 3.0 / 4
        else:
            raise NotImplementedError(f'got unexpected act_func: {act_func}')
    else: # check for instances of modules
        if isinstance(act_func, nn.ReLU):
            return math.sqrt(2.0)
        elif isinstance(act_func, nn.ELU):
            return math.sqrt(2.0)
        elif isinstance(act_func, nn.CELU):
            return math.sqrt(2.0)
        elif isinstance(act_func, nn.GELU):
            return math.sqrt(2.0)
        elif isinstance(act_func, nn.Softplus):
            return math.sqrt(2.0)
        elif isinstance(act_func, nn.LeakyReLU):
            negative_slope = act_func.negative_slope
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        elif isinstance(act_func, nn.Tanh):
            return 5.0 / 3
        elif isinstance(act_func, nn.SELU):
            return 3.0 / 4
        
        # check for functional functions
        if act_func == F.relu:
            return math.sqrt(2.0)
        elif act_func == F.elu:
            return math.sqrt(2.0)
        elif act_func == F.celu:
            return math.sqrt(2.0)
        elif act_func == F.gelu:
            return math.sqrt(2.0)
        elif act_func == F.softplus:
            return math.sqrt(2.0)
        elif act_func == F.leaky_relu:
            negative_slope = 0.1
            return math.sqrt(2.0 / (1 + negative_slope ** 2))
        elif act_func == F.tanh or act_func == torch.tanh:
            return 5.0 / 3
        elif act_func == F.selu:
            return 3.0 / 4
        else:
            raise NotImplementedError(f'got unexpected act_func: {act_func}')


class TransposedConv1dLayer(nnModule):
    def __init__(
            self,
            in_dim, out_dim, cond_dim, kernel_size,
            stride=1,
            groups=1,
            padding=None,
            dilation=1,
            bias=True,
            in_dropout =0., # only one is needed typically
            out_dropout=0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            act_func=None,
            allow_even_kernel_size=False,
            pad_side=None,
            causal_pad=False,
            partialconv_pad=False,
            inplace_dropout=True,
            always_dropout=False,
            w_gain=1.0,
            weight_norm=False,
            spectral_norm=False,
            padding_val=0.0,
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            vector_cond=True,
            slice_cond=False,
            cond_kernel_size=1,
        ):
        super().__init__()
        # common params
        self.  in_dim =   in_dim
        self. out_dim =  out_dim
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        
        # dropout params
        self.  in_dropout_rate = max(  in_dropout, 0.0)
        self. out_dropout_rate = max( out_dropout, 0.0)
        self.cond_dropout_rate = max(cond_dropout, 0.0)
        self.inplace_dropout = inplace_dropout
        self.always_dropout = always_dropout
        self.dropout_func = F.alpha_dropout if act_func in ['selu', 'swish'] else F.dropout 
        
        # get padding
        self.partialconv_pad = partialconv_pad
        self.vector_cond = vector_cond
        self.slice_cond = slice_cond
        self.padding = padding
        self.padding_val = padding_val
        del padding
        if self.padding is None:
            if causal_pad:
                self.padding = (dilation * (kernel_size - 1), 0)
            else:
                assert allow_even_kernel_size or kernel_size % 2 == 1
                left_biased_padding = (math.floor(dilation * (kernel_size - 1) / 2), math.ceil(dilation * (kernel_size - 1) / 2))
                self.padding = left_biased_padding
                if kernel_size % 2 == 0: # if kernel_size is even, check padding side
                    if pad_side == 'left':
                        self.padding = left_biased_padding
                    elif pad_side == 'right':
                        self.padding = left_biased_padding[::-1]
                    else:
                        raise NotImplementedError(f'pad_side of {pad_side} is not supported/expected')
        if isinstance(self.padding, (int, float)):
            self.padding = (self.padding, self.padding)
        
        # get act_func and update weight gain
        self.act_func = None if act_func is None else get_afunc(act_func)
        w_gain = w_gain * get_weight_gain_from_act_func(act_func)
        
        # init conv(s)
        self.conv = nn.ConvTranspose1d(in_dim, out_dim, kernel_size, stride, padding=0, dilation=dilation, bias=bias, groups=groups, output_padding=stride-1)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=w_gain)
        assert bool(weight_norm) + bool(spectral_norm) < 2, 'cannot use any 2 or more of [\'weight_norm\', \'spectral_norm\']'
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv, name='weight')
        
        # add normalization layers
        self.instance_norm = nn.InstanceNorm1d(out_dim,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_dim, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    =       BatchNorm1d(out_dim, affine=affine_norm, momentum=batch_norm_momentum) if    batch_norm else None
        
        # init conditional layer(s)
        if self.cond_dim:
            self.cond_kernel_size = cond_kernel_size
            self.cond_padding = [int(dilation * (cond_kernel_size - 1) / 2), ]*2
            cond_stride = 1 if vector_cond else stride
            self.cond_conv = nn.ConvTranspose1d(
                self.cond_dim, 2*out_dim, kernel_size = cond_kernel_size,
                dilation=1 if vector_cond else dilation,
                stride  =cond_stride,
                output_padding=1,
                padding=0,
            )
            self.cond_conv.weight.data.mul_(1e-3)
            self.cond_conv.bias.data.fill_(0.0)
    
    def reshape_input(self, x, c, m, l):
        assert x.dim() == 3, f'got input dims() = {x.dim()}, expected 3'
        B, T, C = x.shape
        Tout = T*self.stride
        assert C == self.in_dim, f'got input dim of {C}, expected {self.in_dim}'
        x = x.permute(0, 2, 1) # [B, T, C] -> [B, C, T]
        
        if self.cond_dim > 0:
            assert c is not None, f'this layer has cond_dim = {self.cond_dim} but cond is None.'
            c = c.permute(0, 2, 1) # [B, T, C] -> [B, C, T]
            assert c.shape[0] == 1 or c.shape[0] == B, f'got different batch size for x and c, got {B} and {c.shape[0]}'
            assert c.shape[2] == 1 or c.shape[2] == T, f'got different length for x and c, got {T} and {c.shape[2]}'
            assert c.shape[1] == self.cond_dim or (c.shape[1] > self.slice_cond and self.slice_cond), f'got cond.shape[2] = {c.shape[2]}, expected {self.cond_dim}'
        
        if m is not None:
            assert m.shape[0] == 1 or m.shape[0] == B
            assert m.shape[1] == 1 or m.shape[1] == T
            m = m.permute(0, 2, 1) # [B, T, 1] -> [B, 1, T]
        
        if l is not None:
            assert l.shape[0] == 1 or l.shape[0] == B
        return x, c, m, l
    
    def input_dropout(self, x, c):
        if self.in_dropout_rate:
            x = self.dropout_func(x, p=self.in_dropout_rate, inplace=False, training=self.training or self.always_dropout)
        if c is not None and self.cond_dropout_rate:
            c = F.dropout    (c, p=self.cond_dropout_rate, inplace=False, training=self.training or self.always_dropout)
        return x, c
    
    def remove_padding(self, x: Tensor, c: Optional[Tensor]):
        """trim edges of output after t-conv layer"""
        if self.cond_dim:
            if self.vector_cond:
                assert c.shape[1] == 1, f'vector_cond set to True but recieved tensor with length of {c.shape[1]}, shape: {list(c.shape)}'
        return x, c
    
    def maybe_get_mask_from_lengths(self, l, m):
        if l is not None and m is None:
            m = get_mask(l).unsqueeze(1)# [B, 1, T]
        return m
    
    def compute_partialconv_weights(self):
        """compute weights where inputs of ones return ones for non-padded areas"""
        absweight = self.conv.weight.detach().abs()# [out_dim, in_dim/groups, kernel_size]
        pweight = absweight / absweight.sum([1, 2], True).clamp_(min=1e-6)
        
        # nan               = [-1.0, 0.0,  1.0] / [ 0.0] # naive implementation
        # [ 1/2, 0.0,  1/2] = [ 1.0, 0.0,  1.0] / [ 2.0] # with abs
        # [-1/2, 0.0, -1/2] = [-1.0, 0.0, -1.0] / [-2.0] # without abs
        # [ 1/3, 1/3,  1/3] = [ 0.2, 0.2,  0.2] / [ 0.6]
        return pweight
    
    def apply_partialconv_weighting(self, x, c, m, l):
        if self.kernel_size == 1 or m is None: # no-op if kernel_size = 1
            return x, c, m, l
        
        m = self.maybe_get_mask_from_lengths(l, m)
        if self.partialconv_pad and self.kernel_size > 1 and m is not None:
            pweight = self.compute_partialconv_weights()
            x_weight = 1 / F.conv_transpose1d(
                F.pad(m.to(x).expand(x.shape[0], self.in_dim, x.shape[2]), self.padding, value=self.padding_val),
                pweight, stride=self.stride, dilation=self.dilation, groups=self.groups
            )
            # [B, C, T]
            if self.bias:
                x = x.sub_(self.conv.bias[None, :, None]).mul_(x_weight).add_(self.conv.bias[None, :, None])
            else:
                x = x.mul_(x_weight)
        return x, c, m, l
    
    def main(self, x, c, m, l):
        # Conv1d
        x = self.conv(x)[:, :, self.padding[0]:-self.padding[1]]
        
        # (maybe) use partial convolution padding
        x, c, m, l = self.apply_partialconv_weighting(x, c, m, l)
        
        # Normalize
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.batch_norm is not None:
            x = self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Conditional Affine
        if self.cond_dim:
            ct = c[:, :self.cond_dim] if self.slice_cond and c.shape[1] > self.cond_dim else c
            ct = self.cond_conv(ct)
            if self.cond_padding[1] > 0:
                ct = ct[:, :, self.cond_padding[0]:-self.cond_padding[1]]
            else:
                ct = ct[:, :, self.cond_padding[0]:]
            c_scale, c_shift = ct.chunk(2, dim=1)# [B, C, T]
            c_scale.data.add_(1.0)
            x = torch.addcmul(*torch.broadcast_tensors(c_shift, x, c_scale))
            if self.stride > 1 and c.shape[1] != 1:
                c = c[:, :, ::self.stride]
        
        # Activation Function
        if self.act_func is not None:
            x = self.act_func(x)
        
        return x, c, m, l
    
    def reshape_output(self, y: Tensor, c: Optional[Tensor], m: Optional[Tensor], l: Optional[Tensor]):
        y = y.permute(0, 2, 1)# [B, C, T] -> [B, T, C]
        if c is not None:
            c = c.permute(0, 2, 1) # [B, C, T] -> [B, T, C]
        if m is not None:
            m = m.permute(0, 2, 1)# [B, C, T] -> [B, T, C]
        return y, c, m, l
    
    def output_dropout(self, y: Tensor):
        if self.out_dropout_rate:
            y = self.dropout_func(y, p=self.out_dropout_rate, inplace=self.inplace_dropout, training=self.training or self.always_dropout)
        return y
    
    def mask_output(self, y: Tensor, c: Optional[Tensor], m: Optional[Tensor], l: Optional[Tensor]):
        if l is not None:
            # formula taken from https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
            l = (l-1)*self.stride -self.padding[0] -self.padding[1] +self.dilation*(self.kernel_size-1) +1
            l = ((l+1)//2)*2
            
            m = get_mask(l).unsqueeze(1) # [B, 1, T]
            y = y.masked_fill_(~m, 0.0)
        return y, c, m, l
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        x, c, m, l = self.transfer_device([x, c, m, l])
        x, c, m, l = self.reshape_input(x, c, m, l)
        x, c = self.input_dropout(x, c)
        
        y, c, m, l = self.main(x, c, m, l)
        
        y = self.output_dropout(y)
        y, c = self.remove_padding(y, c)
        y, c, m, l = self.mask_output(y, c, m, l)
        y, c, m, l = self.reshape_output(y, c, m, l)
        return y, c, m, l




# Conv1d
# with cond + init techniques
# with multiple act funcs for separable
# with affine scaling applied after normalization and before act-func
# afunc( (norm(x)*weight) + bias )
# instead of current;
# afunc(norm(x))*weight + bias
# with Gated Units available for cond_dim inputs
# with output_shape() method
# with new smoothed batchnorm
# return cond, mask


class Conv1dLayer(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            padding_val=0.0,
            flip_act_func=False,
            slice_cond=False,
            
            # learning capacity kwargs
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # magnitude kwargs
            w_gain=1.0,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            partialconv_pad=False,
        ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        
        # common params
        self.  in_dim =   in_dim
        self. out_dim =  out_dim
        self.cond_dim = cond_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        
        # dropout params
        assert in_dropout <= 1.0, f'in_dropout is greater than 1.0, got {in_dropout}'
        assert out_dropout <= 1.0, f'out_dropout is greater than 1.0, got {out_dropout}'
        assert cond_dropout <= 1.0, f'cond_dropout is greater than 1.0, got {cond_dropout}'
        self.  in_dropout_rate = max(  in_dropout, 0.0)
        self. out_dropout_rate = max( out_dropout, 0.0)
        self.cond_dropout_rate = max(cond_dropout, 0.0)
        self.inplace_dropout = inplace_dropout
        self.always_dropout = always_dropout
        self.dropout_func = F.alpha_dropout if act_func in ['selu', 'swish'] else F.dropout
        
        # get padding
        self.partialconv_pad = partialconv_pad
        self.vector_cond = vector_cond
        self.slice_cond = slice_cond
        self.padding = padding
        self.padding_val = padding_val
        del padding
        if self.padding is None:
            if causal_pad:
                self.padding = (dilation * (kernel_size - 1), 0)
            else:
                assert allow_even_kernel_size or kernel_size % 2 == 1
                left_biased_padding = (math.floor(dilation * (kernel_size - 1) / 2), math.ceil(dilation * (kernel_size - 1) / 2))
                self.padding = left_biased_padding
                if kernel_size % 2 == 0: # if kernel_size is even, check padding side
                    if pad_side == 'left':
                        self.padding = left_biased_padding
                    elif pad_side == 'right':
                        self.padding = left_biased_padding[::-1]
                    else:
                        raise NotImplementedError(f'pad_side of {pad_side} is not supported/expected')
        if isinstance(self.padding, (int, float)):
            self.padding = (self.padding, self.padding)
        
        # get act_func and update weight gain
        self.flip_act_func = flip_act_func
        self.act_func = None if act_func is None else get_afunc(act_func)
        w_gain = w_gain * get_weight_gain_from_act_func(act_func)
        
        # init conv(s)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding=0, dilation=dilation, bias=bias, groups=groups)
        nn.init.xavier_uniform_(self.conv.weight, gain=w_gain)
        
        # param normalization
        assert bool(weight_norm) + bool(spectral_norm) < 2, 'cannot use any 2 or more of [\'weight_norm\', \'spectral_norm\']'
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv, name='weight')
        
        # add normalization layers
        self.instance_norm = nn.InstanceNorm1d(out_dim,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_dim, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    =       BatchNorm1d(out_dim, affine=affine_norm, momentum=batch_norm_momentum) if    batch_norm else None
        
        # init conditional layer(s)
        if self.cond_dim:
            self.cond_kernel_size = cond_kernel_size
            self.cond_padding = [int(dilation * (cond_kernel_size - 1) / 2),]*2
            self.cond_conv = nn.Conv1d(
                self.cond_dim, 2*out_dim,
                kernel_size = cond_kernel_size,
                stride  =1 if vector_cond else stride,
                dilation=1 if vector_cond else dilation,
                padding=0
            )
            self.cond_conv.weight.data.mul_(1e-3)
            self.cond_conv.bias.data.fill_(0.0)
    
    def reshape_input(self, x, c, m, l):
        assert x.dim() == 3, f'got input dims() = {x.dim()}, expected 3'
        B, T, C = x.shape
        assert C == self.in_dim, f'got input dim of {C}, expected {self.in_dim}'
        x = x.permute(0, 2, 1) # [B, T, C] -> [B, C, T]
        
        if self.cond_dim:
            assert c is not None, f'this layer has cond_dim = {self.cond_dim} but cond is None.'
            c = c.permute(0, 2, 1) # [B, T, C] -> [B, C, T]
            assert c.shape[0] == 1 or c.shape[0] == B, f'got different batch size for x and c, got {B} and {c.shape[0]}'
            assert c.shape[2] == 1 or c.shape[2] == T, f'got different length for x and c, got {T} and {c.shape[2]}'
            assert c.shape[1] == self.cond_dim or (c.shape[1] > self.slice_cond and self.slice_cond), f'got cond.shape[2] = {c.shape[2]}, expected {self.cond_dim}'
        
        if m is not None:
            assert m.shape[0] == 1 or m.shape[0] == B
            assert m.shape[1] == 1 or m.shape[1] == T
            m = m.permute(0, 2, 1) # [B, T, 1] -> [B, 1, T]
        
        if l is not None:
            assert l.shape[0] == 1 or l.shape[0] == B
        return x, c, m, l
    
    def input_dropout(self, x, c):
        if self.in_dropout_rate:
            x = self.dropout_func(x, p=self.in_dropout_rate, inplace=False, training=self.training or self.always_dropout)
        if c is not None and self.cond_dropout_rate:
            c = F.dropout(c, p=self.cond_dropout_rate, inplace=False, training=self.training or self.always_dropout)
        return x, c
    
    def handle_padding(self, x: Tensor, c: Optional[Tensor]):
        """pad input tensors before convolution layer"""
        if x is not None:
            x = F.pad(x, self.padding, value=self.padding_val)
        if self.cond_dim:
            if self.vector_cond:
                assert c.shape[2] == 1, f'vector_cond set to True but recieved tensor with length of {c.shape[1]}, shape: {list(c.shape)}'
            c = F.pad(c, self.cond_padding, value=self.padding_val)
        return x, c
    
    def maybe_get_mask_from_lengths(self, l, m):
        if l is not None:
            if m is None:
                m = get_mask(l).unsqueeze(1)  # [B, 1, T]
                m, _ = self.handle_padding(m.transpose(1, 2), None)
                m = m.transpose(1, 2).bool()
        return m
    
    def compute_partialconv_weights(self):
        """compute weights where inputs of ones return ones for non-padded areas"""
        absweight = self.conv.weight.detach().abs()# [out_dim, in_dim/groups, kernel_size]
        pweight = absweight / absweight.sum([1, 2], True).clamp_(min=1e-6)
        
        # nan               = [-1.0, 0.0,  1.0] / [ 0.0] # naive implementation
        # [ 1/2, 0.0,  1/2] = [ 1.0, 0.0,  1.0] / [ 2.0] # with abs
        # [-1/2, 0.0, -1/2] = [-1.0, 0.0, -1.0] / [-2.0] # without abs
        # [ 1/3, 1/3,  1/3] = [ 0.2, 0.2,  0.2] / [ 0.6]
        return pweight
    
    def apply_partialconv_weighting(self, x, c, m, l):
        if self.kernel_size == 1 or m is None: # no-op if kernel_size = 1
            return x, c, m, l
        
        m = self.maybe_get_mask_from_lengths(l, m)
        if self.partialconv_pad and self.kernel_size > 1 and m is not None:
            pweight = self.compute_partialconv_weights()
            x_weight = 1 / F.conv1d(
                F.pad(m.to(x).expand(x.shape[0], self.in_dim, x.shape[2]), self.padding, value=self.padding_val),
                pweight, stride=self.stride, dilation=self.dilation, groups=self.groups
            )
            # [B, C, T]
            if self.bias:
                x = x.sub_(self.conv.bias[None, :, None]).mul_(x_weight).add_(self.conv.bias[None, :, None])
            else:
                x = x.mul_(x_weight)
        return x, c, m, l
    
    def main(self, x, c, m, l):
        # Conv1d
        x = self.conv(x)
        
        # (maybe) use partial convolution padding
        x, c, m, l = self.apply_partialconv_weighting(x, c, m, l)
        
        # Normalize
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.batch_norm is not None:
            x = self.batch_norm(x.permute(0, 2, 1),
                                m.permute(0, 2, 1) if m is not None else None,
                                l.permute(0, 2, 1) if l is not None else None).permute(0, 2, 1)
        
        # Conditional Affine
        if self.cond_dim:
            ct = c[:, :self.cond_dim] if self.slice_cond and c.shape[1] > self.cond_dim else c
            c_scale, c_shift = self.cond_conv(ct).chunk(2, dim=1)# [B, C, T]
            c_scale.data.add_(1.0)
            x = torch.addcmul(*torch.broadcast_tensors(c_shift, x, c_scale))
            if self.stride > 1 and c.shape[1] != 1:
                c = c[:, :, ::self.stride]
        
        # Activation Function
        if self.act_func is not None:
            if self.flip_act_func:
                x = self.act_func(-x).mul_(-1.)
            else:
                x = self.act_func(x)
        
        return x, c, m, l
    
    def reshape_output(self, y: Tensor, c: Optional[Tensor], m: Optional[Tensor], l: Optional[Tensor]):
        y = y.permute(0, 2, 1)# [B, C, T] -> [B, T, C]
        if self.cond_dim:
            c = c.permute(0, 2, 1) # [B, C, T] -> [B, T, C]
        if m is not None:
            m = m.permute(0, 2, 1)# [B, C, T] -> [B, T, C]
        return y, c, m, l
    
    def output_dropout(self, y: Tensor):
        if self.out_dropout_rate and self.act_func is not None:
            # checking for activation function since if activation func is missing, then what is the dropout actually regularizing?
            y = self.dropout_func(y, p=self.out_dropout_rate, inplace=self.inplace_dropout, training=self.training or self.always_dropout)
        return y
    
    def mask_output(self, y: Tensor, c: Optional[Tensor], m: Optional[Tensor], l: Optional[Tensor]):
        m = self.maybe_get_mask_from_lengths(l, m)
        if l is not None:
            # formula taken from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            l = (l + self.padding[0] + self.padding[1] - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        if m is not None:
            #if self.stride != 1 or self.dilation != 1:
            #    m = F.conv1d(
            #        F.pad(m.to(y), self.padding, value=self.padding_val),
            #        m.new_ones((1, 1, self.kernel_size),).to(y)/self.kernel_size,
            #        padding=0,
            #        stride=self.stride,
            #        dilation=self.dilation,
            #    ).clamp_(min=0.0, max=1.0)
            #    m = m >= 1.0
            y = y.masked_fill(~m, self.padding_val)
        return y, c, m, l
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        x, c, m, l = self.transfer_device([x, c, m, l])
        x, c, m, l = self.reshape_input(x, c, m, l)
        x, c = self. input_dropout(x, c)
        x, c = self.handle_padding(x, c)
        
        y, c, m, l = self.main(x, c, m, l)
        
        y = self.output_dropout(y)
        y, c, m, l = self.mask_output(y, c, m, l)
        y, c, m, l = self.reshape_output(y, c, m, l)
        return y, c, m, l


# Conv1dModule
class Conv1dModule(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flip_act_func=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            
            # learning capacity kwargs
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # magnitude kwargs
            w_gain=1.0,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=0,
            separable_hidden_dim=0,
            partialconv_pad=False,
        ):
        super().__init__()
        assert in_dim > 0
        assert out_dim > 0
        assert separable_hidden_dim > 0 or separable < 2
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        
        common_kwargs = dictify(
            bias=bias,
            in_dropout =in_dropout,
            out_dropout=out_dropout,
            cond_dropout=cond_dropout,
            act_func=act_func,
            flip_act_func=flip_act_func,
            allow_even_kernel_size=allow_even_kernel_size,
            pad_side=pad_side,
            causal_pad=causal_pad,
            partialconv_pad=partialconv_pad,
            inplace_dropout=inplace_dropout,
            always_dropout=always_dropout,
            weight_norm=weight_norm,
            spectral_norm=spectral_norm,
            padding_val=padding_val,
            affine_norm=affine_norm,
            instance_norm=instance_norm,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            vector_cond=vector_cond,
            slice_cond=slice_cond,
            cond_kernel_size=cond_kernel_size
        )
        spatial_kwargs = dictify(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        lin_spatial_kwargs = dictify(
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
        )
        
        self.separable = int(separable)
        if self.separable == 2:
            hdn_dim = separable_hidden_dim
            self.conv1 = Conv1dLayer(
                in_dim, hdn_dim, cond_dim if cond_every != 'module' else 0,
                w_gain=w_gain**(1/3), groups=groups,
                **lin_spatial_kwargs, **common_kwargs,
            )
            self.conv2 = Conv1dLayer(
                hdn_dim, hdn_dim, cond_dim,
                w_gain=w_gain**(1/3), groups=hdn_dim,
                **spatial_kwargs, **common_kwargs,
            )
            self.conv3 = Conv1dLayer(
                hdn_dim, out_dim, cond_dim if cond_every != 'module' else 0,
                w_gain=w_gain**(1/3), groups=groups,
                **lin_spatial_kwargs, **common_kwargs,
            )
        elif self.separable:
            self.conv1 = Conv1dLayer(
                in_dim, in_dim, cond_dim if cond_every != 'module' else 0,
                w_gain=w_gain**(1/2), groups=in_dim,
                **spatial_kwargs, **common_kwargs,
            )
            self.conv2 = Conv1dLayer(
                in_dim, out_dim, cond_dim,
                w_gain=w_gain**(1/2), groups=groups,
                **lin_spatial_kwargs, **common_kwargs,
            )
        else:
            self.conv = Conv1dLayer(
                in_dim, out_dim, cond_dim,
                w_gain=w_gain, groups=groups,
                **spatial_kwargs, **common_kwargs,
            )
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        #assert x.isfinite().all(), 'non-finite elements on Conv1dModule input'
        if self.separable == 2:
            x, c, m, l = self.conv1(x, c, m, l)
            x, c, m, l = self.conv2(x, c, m, l)
            y, c, m, l = self.conv3(x, c, m, l)
        elif self.separable:
            x, c, m, l = self.conv1(x, c, m, l)
            y, c, m, l = self.conv2(x, c, m, l)
        else:
            y, c, m, l = self.conv(x, c, m, l)
        #assert y.isfinite().all(), 'non-finite elements on Conv1dModule output'
        return y, c, m, l


# Self-Attention Block
class SelfAttentionLayer(nnModule):
    def __init__(self, hidden_dim, n_heads, dropout,
                 add_pos_embed=True, rel_pos_embed=False, v_pos_embed=True, cat_pos_embed=False,
                 batch_norm=False, instance_norm=False, layer_norm=True,
                 affine_norm=True, batch_norm_momentum=0.1,
                 rezero=False, rezero_grad_mul=0.1, rezero_vector=True, rezero_init_val=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.add_pos_embed = add_pos_embed
        
        if self.add_pos_embed:
            #assert 2 % hidden_dim == 0, ('embed_dim must be even to use pos_embed!')
            self.pos_embed = SinusoidalEmbeddings(dim=hidden_dim, rand_start_pos=rel_pos_embed)
            self.v_pos_embed = v_pos_embed
            self.cat_pos_embed = cat_pos_embed
            if self.cat_pos_embed:
                self.pos_lin = nn.Linear(hidden_dim*2, hidden_dim)
        
        self.rezero = rezero
        if self.rezero:
            self.rezero_grad_mul = rezero_grad_mul
            self.res_weight = nn.Parameter(torch.ones(hidden_dim if rezero_vector else 1)*rezero_init_val)
        
        if n_heads > 0:
            if hidden_dim % n_heads != 0:
                warnings.warn('embed_dim is not divisible by num_heads! ignoring layer')
            else:
                self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=hidden_dim, vdim=hidden_dim, num_heads=n_heads, dropout=dropout)
                
                # add normalization layers
                self.instance_norm = nn.InstanceNorm1d(hidden_dim,             affine=affine_norm) if instance_norm else None
                self.layer_norm    = nn.   LayerNorm  (hidden_dim, elementwise_affine=affine_norm) if    layer_norm else None
                self.batch_norm    =       BatchNorm1d(hidden_dim, affine=affine_norm, momentum=batch_norm_momentum) if batch_norm else None
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        # (maybe) get mask from lengths
        if m is None and l is not None:
            m = get_mask(l).unsqueeze(2)# [B, T, C]
        
        # Position Embedding
        if self.add_pos_embed:
            if self.cat_pos_embed:
                xqk = self.pos_lin(maybe_cat(x, apply_rotary_pos_emb(x.mul(0.0), self.pos_embed(x))))
            else:
                xqk = apply_rotary_pos_emb(x, self.pos_embed(x))
            if self.v_pos_embed:
                x = xqk
        else:
            xqk = x
        
        # Dot-Prod Attention
        if self.n_heads and hasattr(self, 'attention'):
            xqk = xqk.transpose(0, 1)
            xv = x.transpose(0, 1)
            x2 = self.attention(
                xqk, xqk, xqk,
                key_padding_mask= None if m is None else ~m.squeeze(2)
            )[0].transpose(0, 1)
            
            # (if rezero): multiply x2 by res_weight parameter
            if self.rezero:
                gmul = self.rezero_grad_mul
                x2 = ScaleGrad(1/gmul)(ScaleGrad(gmul)(x2) * self.res_weight)
            
            # Add
            x = x + x2
            # and Normalize
            if self.instance_norm is not None:
                x = self.instance_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            if self.layer_norm is not None:
                x = self.layer_norm(x)
            if self.batch_norm is not None:
                x = self.batch_norm(x)
        
        return x, c, m, l


# ResBlock
# contains multiple Conv1d layers
# has a single residual connection
class ResBlock(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            hidden_dim=0,
            n_layers=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            skip_all_res=False,
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_init_val=0.05,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
        ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        assert res_scaling in [None, 'sqrt2', 'rezero', 'bnorm']
        assert res_type in [None, 'slice', '1x1']
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.slice_cond = slice_cond
        
        self.self_attention = SelfAttentionLayer(
            in_dim,
            self_attention_n_heads if use_self_attention else 0,
            dropout=att_dropout,
            add_pos_embed=add_pos_embed,
            rel_pos_embed=rel_pos_embed,
            v_pos_embed  =  v_pos_embed,
            cat_pos_embed=cat_pos_embed,
            batch_norm   =False if self_attention_force_layer_norm else    batch_norm,
            instance_norm=False if self_attention_force_layer_norm else instance_norm,
            layer_norm   =True  if self_attention_force_layer_norm else    layer_norm,
            batch_norm_momentum=batch_norm_momentum,
            rezero = bool(res_scaling == 'rezero'),
            rezero_vector=rezero_vector,
        )
        
        self.res_post_act_func = res_post_act_func
        self.act_func = get_afunc(act_func, return_none=True) if self.res_post_act_func and final_layer_act_func else None
        self.skip_all_res = skip_all_res
        if not self.skip_all_res:
            self.res_type = res_type
            self.res_scaling = res_scaling
            if self.res_scaling == 'rezero':
                self.rezero_grad_mul = rezero_grad_mul
                self.res_weight = nn.Parameter(torch.ones(out_dim if rezero_vector else 1)*rezero_init_val)
            elif self.res_scaling == 'bnorm':
                self.res_bn = BatchNorm1d(out_dim, momentum=None, affine=affine_norm)
            if self.res_type == '1x1':
                self.res_conv = Conv1dLayer(in_dim, out_dim, cond_dim=0, stride=stride**n_layers)
        
        if not hasattr(dilation, '__getitem__'):
            dilation = [dilation, ]
        self.dilation = dilation
        
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        for cur_dilation in dilation:
            for i in range(n_layers):
                is_first_layer = bool(i == 0)
                is_last_layer = bool(i+1 == n_layers)
                self.convs.append(
                    Conv1dModule(
                        in_dim if is_first_layer else hidden_dim,
                        out_dim if is_last_layer else hidden_dim,
                        cond_dim*is_first_layer if cond_every=='resblock' else cond_dim, # if cond_every=='resblock', only use cond on first Module
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=groups,
                        padding=padding,
                        dilation=cur_dilation,
                        bias=bias,
                        in_dropout =in_dropout,
                        out_dropout=out_dropout,
                        cond_dropout=cond_dropout,
                        act_func=act_func if (not final_layer_act_func and not res_post_act_func and not is_last_layer) or final_layer_act_func else None,
                        flip_act_func=i%2==1 if flipping_act_funcs else False,
                        allow_even_kernel_size=allow_even_kernel_size,
                        pad_side=pad_side,
                        causal_pad=causal_pad,
                        partialconv_pad=partialconv_pad,
                        inplace_dropout=inplace_dropout,
                        always_dropout=always_dropout,
                        w_gain=w_gain,
                        weight_norm=weight_norm,
                        spectral_norm=spectral_norm,
                        padding_val=padding_val,
                        affine_norm=affine_norm,
                        instance_norm=instance_norm,
                        layer_norm=layer_norm,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        vector_cond=vector_cond,
                        slice_cond=slice_cond,
                        cond_kernel_size=cond_kernel_size,
                        cond_every='module' if cond_every=='resblock' else cond_every, # options: ['layer', 'module', 'resblock']
                        separable=separable,
                        separable_hidden_dim=separable_hidden_dim,
                    )
                )
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        x, c, m, l = self.transfer_device([x, c, m, l])
        
        # (maybe) self attention layer
        x, c, m, l = self.self_attention(x, c, m, l)
        
        if self.n_layers:
            for j in range(len(self.dilation)):
                x_res = x
                j_convs = self.convs[j*self.n_layers:(j+1)*self.n_layers]
                for conv in j_convs:
                    x, c, m, l = conv(x, c, m, l)
                
                if not self.skip_all_res:
                    if self.res_scaling == 'rezero':
                        x = ScaleGrad(1/self.rezero_grad_mul)(ScaleGrad(self.rezero_grad_mul)(x) * self.res_weight)
                    
                    # add x to residual connection
                    if self.res_type is None:
                        x = x + x_res
                    elif self.res_type == 'slice':
                        if x.shape[2] > x_res.shape[2]:
                            x_res = F.pad(x_res, (0, x.shape[2] - x_res.shape[2]))
                        if x.shape[2] < x_res.shape[2]:
                            x_res = x_res[:, :, :x.shape[2]]
                        x = x + x_res
                    elif self.res_type == '1x1':
                        x = x + self.res_conv(x_res)[0]
                    else:
                        raise NotImplementedError('got res_type of {}, expected any of [None, \'slice\', \'1x1\']')
                    
                    if self.res_scaling == 'sqrt2':
                        x = x / math.sqrt(2)
                    elif self.res_scaling == 'bnorm':
                        x = self.res_bn(x, m, l)
                
                # (optional) run post act_func
                if self.res_post_act_func and self.act_func is not None:
                    x = self.act_func(x)
        return x, c, m, l

# ResNet
# contains multiple ResBlocks
class ResNet(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            skip_all_res=False,
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            ):
        super().__init__()
        if bottleneck_dim is None:
            assert hidden_dim is not None
            bottleneck_dim = hidden_dim
        if out_dim is None:
            assert bottleneck_dim is not None
            out_dim = bottleneck_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.rescale_net_output = rescale_net_output
        self.res_scaling = res_scaling
        self.normalize_input = normalize_input
        self.unnormalize_output = unnormalize_output
        if self.normalize_input:
            self.batchnorm = BatchNorm1d(in_dim, momentum=None, affine=False)
            if unnormalize_output:
                assert in_dim == out_dim, f'in_dim must equal out_dim to use unnormalize_output, got {in_dim} and {out_dim}'
        
        self.use_self_attention = use_self_attention and self_attention_n_heads > 0
        if not add_pos_embed and self.use_self_attention:
            print("warning: using self_attention without add_pos_embed, the self_attention layers may not work correctly.")
        
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            is_first_block = bool(i == 0)
            is_last_block = bool(i+1 == n_blocks)
            self.blocks.append(
                ResBlock(
                    in_dim if is_first_block else bottleneck_dim,
                    out_dim if is_last_block else bottleneck_dim,
                    hidden_dim=hidden_dim,
                    cond_dim=cond_dim*is_first_block if cond_every=='resnet' else cond_dim, # if cond_every=='resnet', only use cond on first ResBlock
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    stride=stride,
                    groups=groups,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    in_dropout =in_dropout, # only one is needed typically
                    att_dropout=att_dropout,
                    out_dropout=out_dropout, # only one is needed typically
                    cond_dropout=cond_dropout, # 0.0 is good
                    act_func=act_func,
                    flipping_act_funcs=flipping_act_funcs,
                    allow_even_kernel_size=allow_even_kernel_size,
                    pad_side=pad_side,
                    causal_pad=causal_pad,
                    partialconv_pad=partialconv_pad,
                    inplace_dropout=inplace_dropout,
                    always_dropout=always_dropout,
                    w_gain=w_gain,
                    weight_norm=weight_norm,
                    spectral_norm=spectral_norm,
                    padding_val=padding_val,
                    affine_norm=affine_norm,
                    instance_norm=instance_norm,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    batch_norm_momentum=batch_norm_momentum,
                    vector_cond=vector_cond,
                    slice_cond=slice_cond,
                    cond_kernel_size=cond_kernel_size,
                    cond_every='resblock' if cond_every=='resnet' else cond_every,
                    separable=separable,
                    separable_hidden_dim=separable_hidden_dim,
                    res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
                    res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero', float]
                    rezero_grad_mul=rezero_grad_mul,
                    rezero_vector=rezero_vector,
                    res_post_act_func=res_post_act_func, # move last act_func to after residual connection
                    final_layer_act_func=(not final_layer_act_func and not is_last_block) or final_layer_act_func, # use activation function on final layer
                    use_self_attention=use_self_attention,
                    self_attention_n_heads=self_attention_n_heads,
                    self_attention_force_layer_norm=self_attention_force_layer_norm,
                    add_pos_embed=add_pos_embed and (is_first_block or not v_pos_embed),
                    rel_pos_embed=rel_pos_embed,
                      v_pos_embed=  v_pos_embed,
                    cat_pos_embed=cat_pos_embed,
                    skip_all_res=skip_all_res,
                )
            )
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        if self.normalize_input:
            x = self.batchnorm(x, m, l)
        if self.n_blocks:
            for block in self.blocks:
                x, c, m, l = block(x, c, m, l)
            if self.rescale_net_output:
                if self.res_scaling is None:
                    x = x / math.sqrt(self.n_blocks)
                if self.use_self_attention:
                    x = x / math.sqrt(2.0)
        if self.normalize_input and self.unnormalize_output:
            x = self.batchnorm.inverse(x, m, l)
        return x, c, m, l


# TransformerBlock


# TransformerNet


# DNNet
# ResNets + strided convs
# low compute load, might bias information towards specific spots
class DNNet(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            skip_all_res=False,
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            dn_scales: List[int] = None,
            pooling_type='avg', # ['max','avg','stridedconv']
        ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.dn_scales = dn_scales
        self.total_down_scale = np.prod(dn_scales)
        self.pooling_type = pooling_type
        
        cur_bottleneck_dim = bottleneck_dim[0] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        cur_hidden_dim     = hidden_dim[0]     if hasattr(hidden_dim,     '__getitem__') else hidden_dim
        cur_cond_dim       = cond_dim[0]       if hasattr(cond_dim,       '__getitem__') else cond_dim
        self.prenet = ResNet(
            in_dim, cur_bottleneck_dim,
            cond_dim=cur_cond_dim,
            hidden_dim=cur_hidden_dim,
            bottleneck_dim=cur_bottleneck_dim,
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stride=stride, groups=groups, padding=padding, dilation=dilation, bias=bias,
            in_dropout=in_dropout, att_dropout=att_dropout, out_dropout=out_dropout, cond_dropout=cond_dropout,
            act_func=act_func, flipping_act_funcs=flipping_act_funcs,
            allow_even_kernel_size=allow_even_kernel_size, pad_side=pad_side,
            causal_pad=causal_pad, partialconv_pad=partialconv_pad,
            inplace_dropout=inplace_dropout, always_dropout=always_dropout, w_gain=w_gain,
            weight_norm=weight_norm, spectral_norm=spectral_norm, padding_val=padding_val,
            affine_norm=affine_norm, instance_norm=instance_norm, layer_norm=layer_norm, batch_norm=batch_norm, batch_norm_momentum=batch_norm_momentum,
            vector_cond=vector_cond, slice_cond=slice_cond, cond_kernel_size=cond_kernel_size, cond_every=cond_every,
            separable=separable,
            separable_hidden_dim=separable_hidden_dim,
            res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=rezero_grad_mul,
            rezero_vector=rezero_vector,
            res_post_act_func=res_post_act_func, # move last act_func to after residual connection
            final_layer_act_func=final_layer_act_func, # use activation function on final layer
            rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
            use_self_attention=use_self_attention,
            self_attention_n_heads=self_attention_n_heads,
            self_attention_force_layer_norm=self_attention_force_layer_norm,
            add_pos_embed=add_pos_embed,
            rel_pos_embed=rel_pos_embed,
              v_pos_embed=  v_pos_embed,
            cat_pos_embed=cat_pos_embed,
        )
        cur_in_dim = self.prenet.out_dim
        
        self.sconv = nn.ModuleList()
        self.scales = nn.ModuleList()
        for i, dn_scale in enumerate(dn_scales):
            is_first_scale = bool(  i==0)
            is_last_scale  = bool(i+1==len(dn_scales))
            cur_bottleneck_dim = bottleneck_dim[i] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
            cur_hidden_dim     = hidden_dim[i]     if hasattr(hidden_dim,     '__getitem__') else hidden_dim
            cur_cond_dim       = cond_dim[i]       if hasattr(cond_dim,       '__getitem__') else cond_dim
            dn_scale = self.dn_scales[i]
            if self.pooling_type == 'stridedconv':
                self.sconv.append(
                    Conv1dModule(
                        cur_in_dim,
                        cur_bottleneck_dim,
                        cur_cond_dim,
                        kernel_size=kernel_size,
                        stride=dn_scale,
                        groups=groups,
                        padding=None,
                        dilation=dilation,
                        bias=bias,
                        in_dropout  =in_dropout, # only one is needed typically
                        out_dropout =out_dropout, # only one is needed typically
                        cond_dropout=cond_dropout, # 0.0 is good
                        act_func=act_func,
                        allow_even_kernel_size=allow_even_kernel_size,
                        pad_side=pad_side,
                        causal_pad=causal_pad,
                        partialconv_pad=partialconv_pad,
                        inplace_dropout=inplace_dropout,
                        always_dropout=always_dropout,
                        w_gain=w_gain,
                        weight_norm=weight_norm,
                        spectral_norm=spectral_norm,
                        padding_val=padding_val,
                        affine_norm=affine_norm,
                        instance_norm=instance_norm,
                        layer_norm=layer_norm,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        vector_cond=vector_cond,
                        slice_cond=slice_cond,
                        cond_kernel_size=cond_kernel_size,
                        cond_every=cond_every,
                        separable=separable,
                        separable_hidden_dim=separable_hidden_dim,
                    )
                )
                cur_in_dim = self.sconv[-1].out_dim
            self.scales.append(
                ResNet(
                    cur_in_dim,
                    out_dim if is_last_scale else cur_bottleneck_dim,
                    cur_cond_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    n_blocks=n_blocks,
                    stride=stride,
                    groups=groups,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    in_dropout  =in_dropout, # only one is needed typically
                    att_dropout =att_dropout,
                    out_dropout =out_dropout, # only one is needed typically
                    cond_dropout=cond_dropout, # 0.0 is good
                    act_func=act_func,
                    allow_even_kernel_size=allow_even_kernel_size,
                    pad_side=pad_side,
                    causal_pad=causal_pad,
                    partialconv_pad=partialconv_pad,
                    inplace_dropout=inplace_dropout,
                    always_dropout=always_dropout,
                    w_gain=w_gain,
                    weight_norm=weight_norm,
                    spectral_norm=spectral_norm,
                    padding_val=padding_val,
                    affine_norm=affine_norm,
                    instance_norm=instance_norm,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    batch_norm_momentum=batch_norm_momentum,
                    vector_cond=vector_cond,
                    slice_cond=slice_cond,
                    cond_kernel_size=cond_kernel_size,
                    separable=separable,
                    separable_hidden_dim=separable_hidden_dim,
                    res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
                    res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
                    res_post_act_func=res_post_act_func, # move last act_func to after residual connection
                    final_layer_act_func=final_layer_act_func, # use activation function on final layer
                    rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
                    use_self_attention=use_self_attention,
                    self_attention_n_heads=self_attention_n_heads,
                    self_attention_force_layer_norm=self_attention_force_layer_norm,
                    add_pos_embed=add_pos_embed,
                    rel_pos_embed=rel_pos_embed,
                      v_pos_embed=  v_pos_embed,
                    cat_pos_embed=cat_pos_embed,
                )
            )
            cur_in_dim = self.scales[-1].out_dim
    
    def downsample(self, i: int, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        dn_scale = self.dn_scales[i]
        if dn_scale == 1:
            return x, c, m, l
        
        if x is not None:
            if self.pooling_type == 'avg':
                x = F.avg_pool1d(x.transpose(1, 2), dn_scale, ceil_mode=True).transpose(1, 2)
            elif self.pooling_type == 'max':
                x = F.max_pool1d(x.transpose(1, 2), dn_scale, ceil_mode=True).transpose(1, 2)
            elif self.pooling_type == 'stridedconv':
                x = self.sconv[i](x, c, m, l)[0]
            else:
                raise NotImplementedError(f'got unexpected pooling_type of {self.pooling_type}')
        
        if c is not None:
            c = F.max_pool1d(c.transpose(1, 2), dn_scale, ceil_mode=True).transpose(1, 2)
        if m is not None:
            m = F.avg_pool1d(m.float().transpose(1, 2), dn_scale, ceil_mode=True).transpose(1, 2).bool()
        if l is not None:
            l = -(-l // dn_scale)
        return x, c, m, l
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        x_list: List[Tensor] = []
        c_list: List[Tensor] = []
        m_list: List[Tensor] = []
        l_list: List[Tensor] = []
        
        x, c, m, l = self.prenet(x, c, m, l)
        x_list.append(x)
        c_list.append(c)
        m_list.append(m)
        l_list.append(l)
        
        for i, scale in enumerate(self.scales):
            x, c, m, l = self.downsample(i, x, c, m, l)
            x, c, m, l = scale(x, c, m, l)
            x_list.append(x)
            c_list.append(c)
            m_list.append(m)
            l_list.append(l)
        return x_list, c_list, m_list, l_list
    
    def infer(self, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        c_list: List[Tensor] = []
        m_list: List[Tensor] = []
        l_list: List[Tensor] = []
        
        c_list.append(c)
        m_list.append(m)
        l_list.append(l)
        
        for i, scale in enumerate(self.scales):
            _, c, m, l = self.downsample(i, None, c, m, l)
            c_list.append(c)
            m_list.append(m)
            l_list.append(l)
        return c_list, m_list, l_list

# UPNet
# ResNets + transposed convs
# low compute load, might bias information towards specific spots where interpolation occurs
class UPNet(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            skip_all_res=False,
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            up_scales: List[int] = None,
            upsample_type='transposedconv', # ['nearest','linear','transposedconv']
        ):
        super().__init__()
        # add top scale
        up_scales      = [1, *up_scales]
        bottleneck_dim = [*bottleneck_dim, bottleneck_dim[-1]] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        hidden_dim     = [*hidden_dim    , hidden_dim    [-1]] if hasattr(hidden_dim,     '__getitem__') else hidden_dim
        cond_dim       = [*cond_dim      , cond_dim      [-1]] if hasattr(cond_dim  ,     '__getitem__') else cond_dim  
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.up_scales = up_scales
        self.total_up_scale = np.prod(up_scales)
        self.upsample_type = upsample_type
        
        cur_in_dim = in_dim
        self.scales = nn.ModuleList()
        for i, dn_scale in enumerate(up_scales):
            is_first_scale = bool(  i==0)
            is_last_scale  = bool(i+1==len(up_scales))
            cur_bottleneck_dim = bottleneck_dim[i] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
            cur_hidden_dim     =     hidden_dim[i] if hasattr(    hidden_dim, '__getitem__') else     hidden_dim
            cur_cond_dim       =       cond_dim[i] if hasattr(      cond_dim, '__getitem__') else       cond_dim
            if not is_last_scale:
                next_bottleneck_dim = bottleneck_dim[i+1] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
            self.scales.append(
                ResNet(
                    cur_in_dim,
                    out_dim if is_last_scale else next_bottleneck_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    cond_dim=cur_cond_dim,
                    kernel_size=kernel_size,
                    n_layers=n_layers,
                    n_blocks=n_blocks,
                    stride=stride,
                    groups=groups,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    in_dropout  =in_dropout, # only one is needed typically
                    att_dropout =att_dropout,
                    out_dropout =out_dropout, # only one is needed typically
                    cond_dropout=cond_dropout, # 0.0 is good
                    act_func=act_func,
                    flipping_act_funcs=flipping_act_funcs,
                    allow_even_kernel_size=allow_even_kernel_size,
                    pad_side=pad_side,
                    causal_pad=causal_pad,
                    partialconv_pad=partialconv_pad,
                    inplace_dropout=inplace_dropout,
                    always_dropout=always_dropout,
                    w_gain=w_gain,
                    weight_norm=weight_norm,
                    spectral_norm=spectral_norm,
                    padding_val=padding_val,
                    affine_norm=affine_norm,
                    instance_norm=instance_norm,
                    layer_norm=layer_norm,
                    batch_norm=batch_norm,
                    batch_norm_momentum=batch_norm_momentum,
                    vector_cond=vector_cond,
                    slice_cond=slice_cond,
                    cond_kernel_size=cond_kernel_size,
                    cond_every=cond_every,
                    separable=separable,
                    separable_hidden_dim=separable_hidden_dim,
                    res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
                    res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
                    rezero_grad_mul=rezero_grad_mul,
                    rezero_vector=rezero_vector,
                    res_post_act_func=res_post_act_func, # move last act_func to after residual connection
                    final_layer_act_func=final_layer_act_func, # use activation function on final layer
                    rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
                    use_self_attention=use_self_attention,
                    self_attention_n_heads=self_attention_n_heads,
                    self_attention_force_layer_norm=self_attention_force_layer_norm,
                    add_pos_embed=add_pos_embed,
                    rel_pos_embed=rel_pos_embed,
                      v_pos_embed=  v_pos_embed,
                    cat_pos_embed=cat_pos_embed,
                )
            )
            cur_in_dim= self.scales[-1].out_dim
        
        self.tconvs = None
        if upsample_type == 'transposedconv':
            self.tconvs = nn.ModuleList()
            for i, dn_scale in enumerate(up_scales):
                is_first_scale = bool(  i==0)
                is_last_scale  = bool(i+1==len(up_scales))
                cur_bottleneck_dim = bottleneck_dim[i] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
                cur_hidden_dim     =     hidden_dim[i] if hasattr(    hidden_dim, '__getitem__') else     hidden_dim
                cur_cond_dim       =       cond_dim[i] if hasattr(      cond_dim, '__getitem__') else       cond_dim
                up_scale = self.up_scales[i]
                self.tconvs.append(
                    TransposedConv1dLayer(
                        cur_bottleneck_dim,
                        out_dim if is_last_scale else cur_bottleneck_dim,
                        cur_cond_dim,
                        kernel_size=kernel_size,
                        stride=up_scale,
                        groups=groups,
                        padding=None,
                        dilation=dilation,
                        bias=bias,
                        in_dropout  =in_dropout, # only one is needed typically
                        out_dropout =out_dropout, # only one is needed typically
                        cond_dropout=cond_dropout, # 0.0 is good
                        act_func=act_func,
                        allow_even_kernel_size=allow_even_kernel_size,
                        pad_side=pad_side,
                        causal_pad=causal_pad,
                        partialconv_pad=partialconv_pad,
                        inplace_dropout=inplace_dropout,
                        always_dropout=always_dropout,
                        w_gain=w_gain,
                        weight_norm=weight_norm,
                        spectral_norm=spectral_norm,
                        padding_val=padding_val,
                        affine_norm=affine_norm,
                        instance_norm=instance_norm,
                        layer_norm=layer_norm,
                        batch_norm=batch_norm,
                        batch_norm_momentum=batch_norm_momentum,
                        vector_cond=vector_cond,
                        slice_cond=slice_cond,
                        cond_kernel_size=cond_kernel_size,
                    )
                )
    
    def upsample(self, i: int, x: Tensor, c: Tensor, m: Tensor, l: Tensor):
        up_scale = self.up_scales[i]
        if self.upsample_type == 'nearest':
            x = F.interpolate(x.transpose(1, 2), scale_factor=up_scale).transpose(1, 2)
        elif self.upsample_type == 'linear':
            x = F.interpolate(x.transpose(1, 2), scale_factor=up_scale, mode='linear', align_corners=False).transpose(1, 2)
        elif self.upsample_type == 'transposedconv':
            c = c[:, :, :self.tconvs[i].cond_dim]
            x = self.tconvs[i](x, c, m, l)[0]
        else:
            raise NotImplementedError(f'got unexpected upsample_type of {self.upsample_type}')
        return x
    
    def forward(self,
                x_in_list: Optional[List[Tensor]],
                c_in_list: Optional[List[Tensor]],
                m_in_list: Optional[List[Tensor]],
                l_in_list: Optional[List[Tensor]],
                reverse_in_lists: bool = True,
        ) -> Tuple[List, List, List, List]:
        if reverse_in_lists:
            x_in_list = x_in_list[::-1]
            c_in_list = c_in_list[::-1]
            m_in_list = m_in_list[::-1]
            l_in_list = l_in_list[::-1]
        
        x_list = []
        c_list = []
        m_list = []
        l_list = []
        
        for i, scale in enumerate(self.scales):
            is_first_scale = bool(i==0)
            
            if is_first_scale:
                x = x_in_list[i]
            else:
                x = self.upsample(i, x, c_in_list[i-1], m_in_list[i-1], l_in_list[i-1])
                x = x[:, :x_in_list[i].shape[1], :] + x_in_list[i]
            
            x, c, m, l = scale(x, c_in_list[i][:, :, :scale.cond_dim], m_in_list[i], l_in_list[i])
            x_list.append(x)
            c_list.append(c)
            m_list.append(m)
            l_list.append(l)
        
        return x_list, c_list, m_list, l_list

# UNet
# DNNet + UPNet with multi-scale residual connections
class UNet(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            up_scales: List[int] = None,
            upsample_type='transposedconv', # ['nearest','linear','transposedconv']
            pooling_type='avg', # ['max','avg','stridedconv']
        ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        common_kwargs = dictify(
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=bias,
            in_dropout  =in_dropout, # only one is needed typically
            att_dropout =att_dropout,
            out_dropout =out_dropout, # only one is needed typically
            cond_dropout=cond_dropout, # 0.0 is good
            act_func=act_func,
            flipping_act_funcs=flipping_act_funcs,
            allow_even_kernel_size=allow_even_kernel_size,
            pad_side=pad_side,
            causal_pad=causal_pad,
            partialconv_pad=partialconv_pad,
            inplace_dropout=inplace_dropout,
            always_dropout=always_dropout,
            w_gain=w_gain,
            weight_norm=weight_norm,
            spectral_norm=spectral_norm,
            padding_val=padding_val,
            affine_norm=affine_norm,
            instance_norm=instance_norm,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            vector_cond=vector_cond,
            slice_cond=slice_cond,
            cond_kernel_size=cond_kernel_size,
            cond_every=cond_every,
            separable=separable,
            separable_hidden_dim=separable_hidden_dim,
            res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=rezero_grad_mul,
            rezero_vector=rezero_vector,
            res_post_act_func=res_post_act_func, # move last act_func to after residual connection
            rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
            normalize_input=normalize_input, # use batchnorm on input features
            unnormalize_output=unnormalize_output, # undo normalize_input on the output side. Only active if normalize_input is True.
            use_self_attention=use_self_attention,
            self_attention_n_heads=self_attention_n_heads,
            self_attention_force_layer_norm=self_attention_force_layer_norm,
            add_pos_embed=add_pos_embed,
            rel_pos_embed=rel_pos_embed,
              v_pos_embed=  v_pos_embed,
            cat_pos_embed=cat_pos_embed,
        )
        
        top_bottleneck_dim = bottleneck_dim[0]    if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        rev_bottleneck_dim = bottleneck_dim[::-1] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        rev_hidden_dim     = hidden_dim    [::-1] if hasattr(hidden_dim,     '__getitem__') else hidden_dim
        rev_cond_dim       = cond_dim      [::-1] if hasattr(cond_dim  ,     '__getitem__') else cond_dim
        self.dnnet = DNNet(
            in_dim, top_bottleneck_dim, rev_cond_dim,
            hidden_dim=rev_hidden_dim,
            bottleneck_dim=rev_bottleneck_dim,
            dn_scales=up_scales[::-1],
            final_layer_act_func=True, # use activation function on final layer
            pooling_type=pooling_type, # ['max','avg','stridedconv']
            **common_kwargs,
        )
        
        in_bottleneck_dim = bottleneck_dim[0] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        self.upnet = UPNet(
            in_bottleneck_dim, out_dim, cond_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            up_scales=up_scales,
            final_layer_act_func=final_layer_act_func,
            upsample_type=upsample_type, # ['nearest','linear','transposedconv']
            **common_kwargs,
        )
    
    def forward(self, x: Tensor, c: Optional[Tensor] = None, m: Optional[Tensor] = None, l: Optional[Tensor] = None):
        x_list, c_list, m_list, l_list = self.dnnet(x, c, m, l)
        x_list, c_list, m_list, l_list = self.upnet(x_list, c_list, m_list, l_list)
        x, c, m, l = x_list[-1], c_list[-1], m_list[-1], l_list[-1]
        return (x, c, m, l), (x_list, c_list, m_list, l_list)


# VDVAEDecoder
class VDVAEDecoder(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            latent_dim=None,
            use_postnet=True,
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            skip_all_res=False,
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            up_scales: List[int] = None,
            upsample_type='nearest', # ['nearest','linear','transposedconv']
            mulogvar_scalar=0.01,
        ):
        super().__init__()
        if latent_dim is None:
            latent_dim = bottleneck_dim
        # add top scale
        up_scales = [1, *up_scales]
        bottleneck_dim = [*bottleneck_dim, bottleneck_dim[-1]] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        hidden_dim     = [*hidden_dim    , hidden_dim    [-1]] if hasattr(hidden_dim,     '__getitem__') else hidden_dim
        cond_dim       = [*cond_dim      , cond_dim      [-1]] if hasattr(cond_dim  ,     '__getitem__') else cond_dim  
        
        assert hasattr(cond_dim, '__getitem__') or cond_dim > 0, 'cond_dim should be greater than 0 for VDVAEDecoder'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.latent_dim = latent_dim
        self.up_scales = up_scales
        self.total_up_scale = np.prod(up_scales)
        self.upsample_type = upsample_type
        
        resnet_kwargs = dictify(
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=bias,
            in_dropout  =in_dropout, # only one is needed typically
            att_dropout =att_dropout,
            out_dropout =out_dropout, # only one is needed typically
            cond_dropout=cond_dropout, # 0.0 is good
            act_func=act_func,
            flipping_act_funcs=flipping_act_funcs,
            allow_even_kernel_size=allow_even_kernel_size,
            pad_side=pad_side,
            causal_pad=causal_pad,
            partialconv_pad=partialconv_pad,
            inplace_dropout=inplace_dropout,
            always_dropout=always_dropout,
            w_gain=w_gain,
            weight_norm=weight_norm,
            spectral_norm=spectral_norm,
            padding_val=padding_val,
            affine_norm=affine_norm,
            instance_norm=instance_norm,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            vector_cond=vector_cond,
            slice_cond=slice_cond,
            cond_kernel_size=cond_kernel_size,
            cond_every=cond_every,
            separable=separable,
            separable_hidden_dim=separable_hidden_dim,
            res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
            res_post_act_func=res_post_act_func, # move last act_func to after residual connection
            rezero_grad_mul=rezero_grad_mul,
            rezero_vector=rezero_vector,
            final_layer_act_func=final_layer_act_func, # use activation function on final layer
            rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
            use_self_attention=use_self_attention,
            self_attention_n_heads=self_attention_n_heads,
            self_attention_force_layer_norm=self_attention_force_layer_norm,
            add_pos_embed=add_pos_embed,
            rel_pos_embed=rel_pos_embed,
              v_pos_embed=  v_pos_embed,
            cat_pos_embed=cat_pos_embed,
        )
        
        self.gtz_enc_list = nn.ModuleList()
        self.prz_enc_list = nn.ModuleList()
        self.postnet_list = nn.ModuleList()
        self.z_dec_list = nn.ModuleList()
        for i, dn_scale in enumerate(up_scales):
            is_first_scale = bool(i==0)
            is_last_scale  = bool(i+1==len(up_scales))
            cur_bottleneck_dim = bottleneck_dim[i] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
            cur_hidden_dim     =     hidden_dim[i] if hasattr(    hidden_dim, '__getitem__') else     hidden_dim
            cur_cond_dim       =       cond_dim[i] if hasattr(      cond_dim, '__getitem__') else       cond_dim
            self.gtz_enc_list.append(
                ResNet(
                    (cur_cond_dim if is_first_scale else prev_bottleneck_dim) + cur_bottleneck_dim,
                    2*latent_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    cond_dim=cur_cond_dim,
                    n_layers=n_layers,
                    n_blocks=n_blocks,
                    **resnet_kwargs,
                )
            )
            self.prz_enc_list.append(
                ResNet(
                    cur_cond_dim if is_first_scale else prev_bottleneck_dim,
                    cur_bottleneck_dim+2*latent_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    cond_dim=cur_cond_dim,
                    n_layers=n_layers,
                    n_blocks=n_blocks,
                    **resnet_kwargs,
                )
            )
            self.z_dec_list.append(
                ResNet(
                    latent_dim, cur_bottleneck_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    cond_dim=cur_cond_dim,
                    n_layers=1,
                    n_blocks=1,
                    **{**resnet_kwargs, 'skip_all_res': True},
                )
            )
            self.postnet_list.append(
                ResNet(
                    cur_bottleneck_dim,
                    out_dim if is_last_scale else cur_bottleneck_dim,
                    hidden_dim=cur_hidden_dim,
                    bottleneck_dim=cur_bottleneck_dim,
                    cond_dim=cur_cond_dim,
                    n_layers=n_layers if use_postnet else (1 if is_last_scale else 0),
                    n_blocks=n_blocks if use_postnet else (1 if is_last_scale else 0), **resnet_kwargs,
                )
            )
            self.mulogvar_scalar = mulogvar_scalar
            prev_bottleneck_dim = cur_bottleneck_dim
    
    def upsample(self, i: int, x: Tensor):
        up_scale = self.up_scales[i]
        if self.upsample_type == 'nearest':
            x = F.interpolate(x.transpose(1, 2), scale_factor=up_scale).transpose(1, 2)
        elif self.upsample_type == 'linear':
            x = F.interpolate(x.transpose(1, 2), scale_factor=up_scale, mode='linear', align_corners=False).transpose(1, 2)
        elif self.upsample_type == 'transposedconv':
            x = self.tconv(x)
        else:
            raise NotImplementedError
        
        return x
    
    def decode_z(self, i: int, xcml: Tuple[Tensor, Tensor, Tensor, Tensor], z_mu: Tensor, z_logvar: Tensor) -> Tensor:
        x, c, m, l = xcml
        z = reparameterize(z_mu, z_logvar, training=1.0)
        return x + self.z_dec_list[i](z, c, m, l)[0]
    
    def forward(self,
                x_in_list: Optional[List[Tensor]],
                c_in_list: Optional[List[Tensor]],
                m_in_list: Optional[List[Tensor]],
                l_in_list: Optional[List[Tensor]],
                reverse_in_lists: bool = True,
        ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        if reverse_in_lists:
            x_in_list = x_in_list[::-1]
            c_in_list = c_in_list[::-1]
            m_in_list = m_in_list[::-1]
            l_in_list = l_in_list[::-1]
        
        x_list = []
        c_list = []
        k_list = []
        m_list = []
        l_list = []
        for i, (gtz_enc, prz_enc, postnet) in enumerate(zip_equal(self.gtz_enc_list, self.prz_enc_list, self.postnet_list)):
            is_first_scale = bool(i==0)
            is_last_scale  = bool(i+1==len(self.gtz_enc_list))
            cur_bottleneck_dim = self.bottleneck_dim[i] if hasattr(self.bottleneck_dim, '__getitem__') else self.bottleneck_dim
            
            if is_first_scale:
                x = c_in_list[i]
            else:
                x = self.upsample(i, x)
            x = x[:, :x_in_list[i].shape[1]]
            z_mulogvar, _, _, _ = gtz_enc(maybe_cat(x, x_in_list[i]), c_in_list[i], m_in_list[i], l_in_list[i])
            z_mulogvar = z_mulogvar * self.mulogvar_scalar
            zt_mu, zt_logvar = z_mulogvar.chunk(2, dim=2) 
            
            x, c, m, l = prz_enc(x, c_in_list[i], m_in_list[i], l_in_list[i])
            x, zp_mu, zp_logvar = x.split([cur_bottleneck_dim, self.latent_dim, self.latent_dim], dim=2)
            zp_mu, zp_logvar = zp_mu*self.mulogvar_scalar, zp_logvar*self.mulogvar_scalar
            
            x = self.decode_z(i, (x, c, m, l), zt_mu, zt_logvar)
            x, c, m, l = postnet(x, c, m, l)
            
            x_list.append(x)
            c_list.append(c)
            k_list.append(kld_loss(zt_mu, zt_logvar, zp_mu, zp_logvar, normal_weight=0.01))
            m_list.append(m)
            l_list.append(l)
        
        return x_list, c_list, k_list, m_list, l_list
    
    def infer(self,
                c_in_list: Optional[List[Tensor]],
                m_in_list: Optional[List[Tensor]],
                l_in_list: Optional[List[Tensor]],
                reverse_in_lists: bool = True,
        ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        if reverse_in_lists:
            c_in_list = c_in_list[::-1]
            m_in_list = m_in_list[::-1]
            l_in_list = l_in_list[::-1]
        
        x_list = []
        c_list = []
        m_list = []
        l_list = []
        for i, (prz_enc, postnet) in enumerate(zip_equal(self.prz_enc_list, self.postnet_list)):
            is_first_scale = bool(i==0)
            is_last_scale  = bool(i+1==len(self.gtz_enc_list))
            cur_bottleneck_dim = self.bottleneck_dim[i] if hasattr(self.bottleneck_dim, '__getitem__') else self.bottleneck_dim
            
            if is_first_scale:
                x = c_in_list[i]
            else:
                x = self.upsample(i, x)
            x = x[:, :c_in_list[i].shape[1]]
            
            x, c, m, l = prz_enc(x, c_in_list[i], m_in_list[i], l_in_list[i])
            x, zp_mu, zp_logvar = x.split([cur_bottleneck_dim, self.latent_dim, self.latent_dim], dim=2)
            zp_mu, zp_logvar = zp_mu*self.mulogvar_scalar, zp_logvar*self.mulogvar_scalar
            
            x = self.decode_z(i, (x, c, m, l), zp_mu, zp_logvar)
            x, c, m, l = postnet(x, c, m, l)
            
            x_list.append(x)
            c_list.append(c)
            m_list.append(m)
            l_list.append(l)
        
        return x_list, c_list, m_list, l_list

# VDVAE
class VDVAE(nnModule):
    def __init__(
            self,
            # I/O dims
            in_dim,
            out_dim,
            cond_dim,
            
            # misc
            flipping_act_funcs=False,
            padding_val=0.0,
            cond_every='layer', # options: ['layer', 'module', 'resblock']
            slice_cond=False,
            res_post_act_func=False, # move last act_func to after residual connection
            final_layer_act_func=False, # use activation function on final layer
            
            # learning capacity kwargs
            latent_dim=None,
            use_postnet=True,
            bottleneck_dim=0,
            hidden_dim=0,
            n_layers=0,
            n_blocks=0,
            groups=1,
            bias=True,
            act_func=None,
            output_act_func=None,
            
            # weight normalization kwargs
            weight_norm=False,
            spectral_norm=False,
            
            # latent normalization kwargs
            affine_norm=True,
            instance_norm=False,
            layer_norm=False,
            batch_norm=False,
            batch_norm_momentum=0.1,
            
            # residual kwargs
            res_type='slice', # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=None, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=0.1,
            rezero_vector=True,
            
            # magnitude kwargs
            w_gain=1.0,
            rescale_net_output=True,
            normalize_input=False,
            unnormalize_output=True,
            
            # regularization kwargs
            in_dropout  =0., # only one is needed typically
            att_dropout =0.,
            out_dropout =0., # only one is needed typically
            cond_dropout=0., # 0.0 is good
            inplace_dropout=True,
            always_dropout=False,
            
            # spatial kwargs
            kernel_size=1,
            allow_even_kernel_size=False,
            pad_side=None,
            stride=1,
            padding=None,
            dilation: Union[List, int] = 1,
            causal_pad=False,
            vector_cond=True,
            cond_kernel_size=1,
            separable=False,
            separable_hidden_dim=0,
            partialconv_pad=False,
            
            add_pos_embed=False,
            rel_pos_embed=False,
              v_pos_embed=True,
            cat_pos_embed=False,
            use_self_attention=False,
            self_attention_n_heads=2,
            self_attention_force_layer_norm=False,
            up_scales: List[int] = None,
            upsample_type='nearest', # ['nearest','linear','transposedconv']
            pooling_type='avg', # ['max','avg','stridedconv']
        ):
        super().__init__()
        common_kwargs = dictify(
            kernel_size=kernel_size,
            n_layers=n_layers,
            n_blocks=n_blocks,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=bias,
            in_dropout  =in_dropout, # only one is needed typically
            att_dropout =att_dropout,
            out_dropout =out_dropout, # only one is needed typically
            cond_dropout=cond_dropout, # 0.0 is good
            act_func=act_func,
            flipping_act_funcs=flipping_act_funcs,
            final_layer_act_func=final_layer_act_func, # use activation function on final layer
            allow_even_kernel_size=allow_even_kernel_size,
            pad_side=pad_side,
            causal_pad=causal_pad,
            partialconv_pad=partialconv_pad,
            inplace_dropout=inplace_dropout,
            always_dropout=always_dropout,
            w_gain=w_gain,
            weight_norm=weight_norm,
            spectral_norm=spectral_norm,
            padding_val=padding_val,
            affine_norm=affine_norm,
            instance_norm=instance_norm,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            vector_cond=vector_cond,
            slice_cond=slice_cond,
            cond_kernel_size=cond_kernel_size,
            cond_every=cond_every,
            separable=separable,
            separable_hidden_dim=separable_hidden_dim,
            res_type=res_type, # options: [None, 'slice', '1x1'] # [change nothing, pad/slice to match, use 1x1 conv to match shape]
            res_scaling=res_scaling, # options: [None, 'sqrt2', 'rezero']
            rezero_grad_mul=rezero_grad_mul,
            rezero_vector=rezero_vector,
            res_post_act_func=res_post_act_func, # move last act_func to after residual connection
            rescale_net_output=rescale_net_output, # divide output by sqrt(n_blocks) or sqrt(2)
            normalize_input=normalize_input, # use batchnorm on input features
            unnormalize_output=unnormalize_output, # undo normalize_input on the output side. Only active if normalize_input is True.
            use_self_attention=use_self_attention,
            self_attention_n_heads=self_attention_n_heads,
            self_attention_force_layer_norm=self_attention_force_layer_norm,
            add_pos_embed=add_pos_embed,
            rel_pos_embed=rel_pos_embed,
              v_pos_embed=  v_pos_embed,
            cat_pos_embed=cat_pos_embed,
        )
        
        top_bottleneck_dim = bottleneck_dim[0]    if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        rev_bottleneck_dim = bottleneck_dim[::-1] if hasattr(bottleneck_dim, '__getitem__') else bottleneck_dim
        rev_hidden_dim     = hidden_dim    [::-1] if hasattr(hidden_dim,     '__getitem__') else hidden_dim
        rev_cond_dim       = cond_dim      [::-1] if hasattr(cond_dim  ,     '__getitem__') else cond_dim
        self.dnnet = DNNet(
            in_dim, top_bottleneck_dim,
            hidden_dim=rev_hidden_dim,
            bottleneck_dim=rev_bottleneck_dim,
            cond_dim=rev_cond_dim,
            dn_scales=up_scales[::-1],
            pooling_type=pooling_type, # ['max','avg','stridedconv']
        )
        
        self.vddec = VDVAEDecoder(
            top_bottleneck_dim, out_dim, cond_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            latent_dim=latent_dim,
            up_scales=up_scales,
            use_postnet=use_postnet,
            upsample_type=upsample_type, # ['nearest','linear','transposedconv']
        )
        
        self.output_act_func = output_act_func
        self.act_func = get_afunc(output_act_func, return_none=True)
    
    def get_kld_loss(self, k_list, m_list):
        if any(type(m) is Tensor for m in m_list):
            m_look = {m.shape[1]: m for m in m_list}
            
            m = m_look[k_list[0].shape[1]]
            n_elems = m.sum(dtype=torch.float)
            kld = k_list[0].masked_fill(~m, 0.0).sum([1, 2], dtype=torch.float)
            
            for k in k_list[1:]:
                m = m_look[k.shape[1]]
                n_elems += m.sum()
                kld += k.masked_fill(~m, 0.0).sum([1, 2], dtype=torch.float)
        else:
            n_elems = k_list[0].numel()
            kld = k_list[0].sum(dtype=torch.float)
            for k in k_list[1:]:
                n_elems += k_list[0].numel()
                kld += k.sum([1, 2], dtype=torch.float)
        return kld / n_elems
    
    def forward(self, x: Tensor, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        x_list, c_list, m_list, l_list = self.dnnet(x, c, m, l)
        x_list, c_list, k_list, m_list, l_list = self.vddec(x_list, c_list, m_list, l_list)
        x, c, m, l = x_list[-1], c_list[-1], m_list[-1], l_list[-1]
        mkld = self.get_kld_loss(k_list, m_list)
        if self.act_func:
            x = self.act_func(x)
        return (x, c, mkld, m, l), (x_list, c_list, k_list, m_list, l_list)
    
    def infer(self, c: Optional[Tensor]=None, m: Optional[Tensor]=None, l: Optional[Tensor]=None):
        if m is None and l is not None:
            m = get_mask1d(l)
        c_list, m_list, l_list = self.dnnet.infer(c, m, l)
        x_list, c_list, m_list, l_list = self.vddec.infer(c_list, m_list, l_list)
        x, c, m, l = x_list[-1], c_list[-1], m_list[-1], l_list[-1]
        if self.act_func:
            x = self.act_func(x)
        return (x, c, m, l), (x_list, c_list, m_list, l_list)

# DilatedBlock
# stack of ResBlock with exponentially increasing dilation


# DilatedNet
# stack of DilatedBlock with skip connections from each ResBlock to final output
# and single post ResNet for output


if __name__ == '__main__':
    def test_conv():
        layer = Conv1dLayer(160, 160, 0, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert y.shape == x.shape
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
        
        
        layer = Conv1dLayer(160, 160, 0, kernel_size=3, dilation=3, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert y.shape == x.shape
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
        
        
        layer = Conv1dLayer(160, 256, 0, kernel_size=3, dilation=3, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert list(y.shape) == [*x.shape[:2], 256]
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
    
    def test_sep_conv():
        layer = Conv1dModule(160, 160, 0, separable=1, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert y.shape == x.shape
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
        
        
        layer = Conv1dModule(160, 256, 0, separable=1, kernel_size=3, dilation=3, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert list(y.shape) == [*x.shape[:2], 256]
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
        
        
        layer = Conv1dModule(160, 160, 0, separable=2, separable_hidden_dim=1024, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert y.shape == x.shape
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
        
        
        layer = Conv1dModule(160, 256, 0, separable=2, separable_hidden_dim=1024, kernel_size=3, dilation=3, act_func='relu')
        x = torch.randn(1, 800, 160)
        y = layer(x)[0]
        assert y.min() == 0.0
        assert list(y.shape) == [*x.shape[:2], 256]
        
        y.sum().backward()
        assert all(x.grad is not None for x in layer.parameters())
    
    #
    # Conv1dLayer
    #   Conv1dModule
    #       ResBlock
    #       SelfAttBlock
    #           ResNet
    #               DNNet
    #               UPNet
    #                   UNet
    #               VDVAEDecoder
    #               VDVAEEncoder
    #                   VDVAE
    #
    
    # Test Conv
    test_conv()
    
    # Test Separable Conv + MBConv
    test_sep_conv()
    
    # Test Sine Pos Embed
    
    # Test Rotary Pos Embed
    
    # Test Learned Pos Embed
    
    # Test ResBlock normal
    
    # Test ResBlock rezero
    
    # Test ResBlock 1x1
    
    # Test ResBlock bnorm
    
    # Test Transformer Preset
    
    # Test Conformer Preset
    
    # Test Dilated WaveNet Preset
    
    # Test UNet Preset
    
    # Test VDVAE Preset (DNNet + stacked TopDown VAE Layers)
    
    # Test Tacotron Encoder Preset (3x Conv1d + 1x LSTM)
    
    def x(): pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

from CookieSpeech.utils.misc_utils import zip_equal

from CookieSpeech.utils.modules.activation_funcs import get_afunc, get_afunc_gain
from torch import Tensor
from typing import List, Tuple, Optional, Union

from CookieSpeech.utils.modules.core import nnModule
from CookieSpeech.utils.modules.utils import get_mask


class ConvNorm2d(nnModule):
    def __init__(self,
                 in_channels, out_channels, kernel_size=1, stride=1, groups=None, padding=None, dilation=1, bias=True, dropout=0.,
                 w_init_gain=None, act_func=None, act_func_params=None, separable=False,
                 causal=False, pad_right=False, partial_padding=False, ignore_separable_warning=False,
                 LSUV_init=False, n_LSUV_passes=1, LSUV_ignore_act_func=False, LSUV_init_bias=False, w_gain=1.0,
                 weight_norm=False, spectral_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False,
                 pad_value=0.0, alpha_dropout=None, always_dropout=False):
        super().__init__()
        if act_func_params is None:
            act_func_params = {}
        if separable and (not ignore_separable_warning):
            assert out_channels//in_channels==out_channels/in_channels, "in_channels must be equal to or a factor of out_channels to use separable Conv1d."
        if not ignore_norm_warning:
            assert bool(instance_norm)+bool(layer_norm)+bool(batch_norm) <= 1, 'only one of instance_norm, layer_norm or batch_norm is recommended to be used at a time. Use ignore_norm_warning=True if you know what you\'re doing'
        if act_func is not None and bias is False:
            print("Warning! Using act_func without any layer bias")
        assert not (spectral_norm and weight_norm), 'can\'t use weight_norm and spectral_norm at the same time'
        self.instance_norm = nn.InstanceNorm2d(out_channels,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_channels, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    = nn.   BatchNorm2d(out_channels,             affine=affine_norm) if    batch_norm else None
        
        self.partial_padding = partial_padding
        self.pad_value = pad_value
        self.weight_norm = weight_norm
        self.act_func    = act_func
        if type(self.act_func) is str:
            w_init_gain = w_init_gain or get_afunc_gain(self.act_func)[0]
            self.act_func = get_afunc(self.act_func)
        self.act_func_params = act_func_params
        
        dilation = self.init_dilation(dilation)
        
        if padding is None:
            if type(kernel_size) in [int]:
                assert(kernel_size % 2 == 1)
                kernel_size = [kernel_size, kernel_size]
            
            if type(kernel_size) in [list, tuple]:
                assert all((ks % 2 == 1) for ks in kernel_size), f'got even number in kernel_size. kernel_size = {kernel_size}'
                padding = [int(d * (k - 1) / 2) for d, k in zip_equal(dilation, kernel_size)]
        
        if type(stride) in [int]:
            stride = [stride, stride]
        self.stride = stride
        
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.separable = (separable and in_channels==out_channels)
        
        self.dropout = dropout
        self.dropout_func = F.alpha_dropout if alpha_dropout or (alpha_dropout is None and act_func == 'selu') else F.dropout
        self.always_dropout = always_dropout
        
        conv_groups = groups or (min(in_channels, out_channels) if self.separable else 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=0 if causal else padding,
                              dilation=dilation, bias=bias, groups=conv_groups)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=w_gain * torch.nn.init.calculate_gain('linear' if self.separable else (w_init_gain or 'linear')))
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv, name='weight')
        
        if self.separable:
            self.conv_d = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias)
            torch.nn.init.xavier_uniform_(
                self.conv_d.weight, gain=torch.nn.init.calculate_gain(w_init_gain or 'linear'))
            if weight_norm:
                self.conv_d = nn.utils.weight_norm(self.conv_d, name='weight')
            if spectral_norm:
                self.conv_d = nn.utils.spectral_norm(self.conv_d, name='weight')
        
        self.causal_pad = (kernel_size-stride)*dilation if causal else 0
        self.pad_right  = pad_right
        
        if LSUV_init:
            self.LSUV_bias = LSUV_init_bias
            self.LSUV_ignore_act_func = LSUV_ignore_act_func
            self.n_LSUV_passes = n_LSUV_passes
            self.register_buffer('LSUV_init_done', torch.tensor(False))
        self.squeeze_t_dim = False
    
    def init_dilation(self, dilation):
        if dilation is None:
            dilation = [1, 1]
        if type(dilation) is int:
            dilation = [dilation, ] * 2
        if len(dilation) == 1:
            dilation = [*dilation, *dilation]
        return dilation
    
    def maybe_pad(self, signal, pad_right=False):
        if self.causal_pad:
            if pad_right:
                signal = F.pad(signal, (0, self.causal_pad))
            else:
                signal = F.pad(signal, (self.causal_pad, 0))
        return signal
    
    def pre(self, signal):# [B, T1, T2, C] or [B, 1, 1, C]
        assert len(signal.shape) == 4, f"input has {len(signal.shape)} dims, should have 4 for Conv2d"
        signal = signal.permute(0, 3, 1, 2)# [B, T1, T2, C] -> [B, C, T1, T2]
        
        assert signal.shape[1] == self.in_channels, f"input has {signal.shape[1]} channels but expected {self.in_channels}. input.shape = {list(signal.shape)}"
        signal = self.maybe_pad(signal, self.pad_right)
        return signal# [B, C, T1, T2]
    
    def main(self, signal, ignore_norm=False, ignore_act_func=False, mask=None):# [B, C, T1, T2]
        #assert torch.isfinite(signal).all(), 'got non-finite element in input!'
        conv_signal = self.conv(signal)
        if self.separable:
            conv_signal = self.conv_d(conv_signal)
        #assert torch.isfinite(signal).all(), 'got non-finite element in output!'
        
        if self.partial_padding and self.padding:
            # multiply values near the edge by (total edge n_elements/non-padded edge n_elements)
            pad = self.padding
            if mask is None:
                mask = signal.abs().sum(1, True)!=self.pad_value# read zeros in input as masked timesteps
                # [B, 1, T1, T2]
            signal_divisor = F.conv1d(mask.to(signal), signal.new_ones((1, 1, *self.kernel_size),)/prod(self.kernel_size), padding=pad, stride=self.stride, dilation=self.dilation).clamp_(min=0.0, max=1.0).masked_fill_(~mask[:, :, ::self.stride[0], ::self.stride[1]], 1.0)
            
            if self.conv.bias is not None:
                bias = self.conv.bias.view(1, self.out_channels, 1, 1)# [1, oC, 1, 1]
                conv_signal = conv_signal.sub_(bias).div(signal_divisor).add_(bias).masked_fill_(~mask[:, :, ::self.stride[0], ::self.stride[1]], self.pad_value)
            else:
                conv_signal = conv_signal.div(signal_divisor).masked_fill_(~mask[:, :, ::self.stride[0], ::self.stride[1]], self.pad_value)
        
        if not ignore_norm:
            conv_signal = self.instance_norm(conv_signal) if self.instance_norm is not None else conv_signal
            conv_signal = self.   batch_norm(conv_signal) if self.   batch_norm is not None else conv_signal
            conv_signal = self.   layer_norm(conv_signal
                        .transpose(1, 2)).transpose(1, 2) if self.   layer_norm is not None else conv_signal
        if self.act_func is not None and not ignore_act_func:
            conv_signal = self.act_func(conv_signal, **self.act_func_params)
        if (self.training or self.always_dropout) and self.dropout > 0.:
            conv_signal = self.dropout_func(conv_signal, p=self.dropout, training=True)
        
        return conv_signal.permute(0, 2, 3, 1)# [B, C, T1, T2] -> [B, T1, T2, C]
    
    def maybe_UV_init(self, signal):
        if hasattr(self, 'LSUV_init_done') and not self.LSUV_init_done and self.training:
            orig_dropout = self.dropout
            self.dropout = 0.0
            for i in range(self.n_LSUV_passes):
                with torch.no_grad():
                    if self.separable:
                        y = self.conv(signal)
                        self.conv.weight.data /= y.std()
                        if hasattr(self.conv, 'bias') and self.LSUV_bias: self.conv.bias.data -= y.mean()
                    
                        y = self.main(signal, ignore_norm=True, ignore_act_func=self.LSUV_ignore_act_func)
                        self.conv_d.weight.data /= y.std()
                        if hasattr(self.conv_d, 'bias') and self.LSUV_bias: self.conv_d.bias.data -= y.mean()
                    else:
                        y = self.main(signal, ignore_norm=True, ignore_act_func=self.LSUV_ignore_act_func)
                        self.conv.weight.data /= y.std()
                        if hasattr(self.conv, 'bias') and self.LSUV_bias: self.conv.bias.data -= y.mean()
            del y
            self.dropout = orig_dropout
            self.LSUV_init_done += True
    
    def forward(self, signal):# [B, T1, T2, C]
        signal = self.pre(signal)# -> [B, C, T1, T2]
        
        self.maybe_UV_init(signal)
        
        conv_signal = self.main(signal)# -> [B, T1, T2, C]
        return conv_signal# [B, T1, T2, C]


class CondConv2d(ConvNorm2d):# Conditional Conv Norm
    def __init__(self, *args, cond_dim=0, mask_zeros=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_zeros = mask_zeros
        if cond_dim:
            out_channels = args[1] if len(args) > 1 else kwargs.get('out_channels')
            self.affine = ConvNorm2d(cond_dim, 2*out_channels)
            self.affine.conv.weight.data *= 0.001
            self.affine.conv.bias.data *= 0.00
    
    def forward(self, signal, cond=None, mask=None):# [B, T1, T2, C], [B, T1, T2, C], [B, T1, T2, 1]
        if self.mask_zeros and mask is None:
            mask = signal.abs().sum(3, keepdim=True) != 0.0
        
        conv_signal = super().forward(signal)# [B, T1, T2, C]
        
        if hasattr(self, 'affine'):
            assert cond is not None
            assert cond.shape[1] == conv_signal.shape[1] or cond.shape[1] == 1, f'got length1 of {conv_signal.shape[1]}, expected {cond.shape[1]} . cond.shape = {cond.shape}'
            assert cond.shape[2] == conv_signal.shape[2] or cond.shape[2] == 1, f'got length2 of {conv_signal.shape[2]}, expected {cond.shape[2]} . cond.shape = {cond.shape}'
            scale_sub1, bias = self.affine(cond).chunk(2, dim=3)# -> [B, T1, T2, C] *2
            conv_signal = torch.addcmul(bias, conv_signal, scale_sub1+1.0)# (conv_signal*scale)+bias
        
        if self.mask_zeros:
            assert conv_signal.shape[1] == mask.shape[1], f'got length1 of {mask.shape[1]}, expected {conv_signal.shape[1]} . mask.shape = {mask.shape}'
            assert conv_signal.shape[2] == mask.shape[2], f'got length2 of {mask.shape[2]}, expected {conv_signal.shape[2]} . mask.shape = {mask.shape}'
            conv_signal = conv_signal.masked_fill_(~mask, 0.0)
        return conv_signal

if __name__ == '__main__':
    # basic Conv2d init + output shape tests
    
    # in_channels, out_channels, kernel_size=1, stride=1, groups=None, padding=None, dilation=1, bias=True, dropout=0.,
    # w_init_gain=None, act_func=None, act_func_params=None, separable=False,
    # causal=False, pad_right=False, partial_padding=False, ignore_separable_warning=False,
    # LSUV_init=False, n_LSUV_passes=1, LSUV_ignore_act_func=False, LSUV_init_bias=False, w_gain=1.0,
    # weight_norm=False, spectral_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False,
    # pad_value=0.0, alpha_dropout=None, always_dropout=False
    
    layer = ConvNorm2d(1, 64)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=3)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=(3, 1))
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=(3, 1), stride=[1, 3])
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 22, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=(3, 1), stride=[1, 1], padding=[0, 0])
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 62, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=(5, 1), stride=[1, 1], padding=[0, 0])
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 60, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, act_func='relu')
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check LSUV
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, LSUV_init=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check weight_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, weight_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check spectral_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, spectral_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check partial_padding
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, partial_padding=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check instance_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, instance_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check layer_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, layer_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check batch_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, batch_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check affine_norm
    layer = ConvNorm2d(1, 64, kernel_size=3, separable=True, affine_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    output = layer(x_inp)
    assert output.shape == (1, 64, 64, 64)
    
    # check CondConv2d
    layer = CondConv2d(1, 64, kernel_size=3, cond_dim=32, separable=True, affine_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    cond  = torch.rand(1, 64, 64, 32)# [B, H, W, C]
    output = layer(x_inp, cond=cond)
    assert output.shape == (1, 64, 64, 64)
    
    # check mask
    layer = CondConv2d(1, 64, kernel_size=3, cond_dim=32, separable=True, affine_norm=True)
    x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
    cond  = torch.rand(1, 64, 64, 32)# [B, H, W, C]
    mask  = get_mask(torch.tensor([64]))[:, :, None, None] & get_mask(torch.tensor([64]))[:, None, :, None]
    output = layer(x_inp, cond=cond, mask=mask)
    assert output.shape == (1, 64, 64, 64)
    
    # check act_funcs
    for act_func in [None, 'ReLU', 'LeakyReLU', 'SELU', 'ELU', 'CELU', 'GELU', 'Softplus']:
        try:
            layer = CondConv2d(1, 64, kernel_size=3, cond_dim=32, separable=True, affine_norm=True, act_func=act_func)
            x_inp = torch.rand(1, 64, 64, 1)# [B, H, W, C]
            cond  = torch.rand(1, 64, 64, 32)# [B, H, W, C]
            mask  = get_mask(torch.tensor([64]))[:, :, None, None] & get_mask(torch.tensor([64]))[:, None, :, None]
            output = layer(x_inp, cond=cond, mask=mask)
            assert output.shape == (1, 64, 64, 64)
        except Exception as ex:
            print(f'act_func = {act_func} failed')
            raise ex
    
    print("TEST COMPLETE!\n"
          "output shapes are correct and no exceptions were rasied during initialization or forward.")

"""
MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Modules are taken from https://github.com/jik876/hifi-gan
# and rewritten as required to work with my codes config files
import random
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod

from CookieSpeech.utils.modules.embeddings import SpeakerEmbedding
from torch.nn import Conv2d
from torch.nn.utils import weight_norm, spectral_norm

from CookieSpeech.utils.modules.core import ConvNorm, nnModule, ConvTranspose, CondConv
from CookieSpeech.utils.modules.core2d import CondConv2d
from CookieSpeech.utils.modules.utils import avg_pool1d, get_mask1d, Fpad

D_LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class DiscriminatorP(nn.Module):
    def __init__(self, period, input_dim=1, cond_dim=0, kernel_size=5, stride=3, use_spectral_norm=False, init_channels=32, groups=1):
        super(DiscriminatorP, self).__init__()
        self.stride = stride
        self.period = period
        conv_kwargs = {'cond_dim': cond_dim, 'groups': groups, 'weight_norm': not use_spectral_norm, 'spectral_norm': use_spectral_norm, 'act_func': nn.LeakyReLU(D_LRELU_SLOPE)}
        self.convs = nn.ModuleList([
            CondConv2d(       input_dim,    init_channels, (kernel_size, 1), (stride, 1), **conv_kwargs),
            CondConv2d(   init_channels,  4*init_channels, (kernel_size, 1), (stride, 1), **conv_kwargs),
            CondConv2d( 4*init_channels, 16*init_channels, (kernel_size, 1), (stride, 1), **conv_kwargs),
            CondConv2d(16*init_channels, 32*init_channels, (kernel_size, 1), (stride, 1), **conv_kwargs),
            CondConv2d(32*init_channels, 32*init_channels, (kernel_size, 1), (     1, 1), **conv_kwargs),
        ])
        self.conv_post = CondConv2d(32*init_channels, 1, (3, 1), cond_dim=cond_dim)
    
    def shape_1d_to_2d(self, x):# [B, T, C] -> [B, T1, T2, C]
        # 1d to 2d
        B, T, C = x.shape
        if T % self.period != 0: # pad first
            n_pad = self.period - (T % self.period)
            x = Fpad(x, (0, n_pad), "reflect")
            T = T + n_pad
        return x.view(B, T // self.period, self.period, C)
    
    def forward(self, x, cond=None):
        fmap = []
        
        x = self.shape_1d_to_2d(x)# [B, T, C] -> [B, T//period, period, C]
        if cond is not None:
            cond = cond.unsqueeze(1)# [B, 1, C] -> [B, 1, 1, C]
        for i, l in enumerate(self.convs):
            x = l(x, cond=cond)
            fmap.append(x)
        x = self.conv_post(x, cond=cond)
        
        # [B, Ti//period, period, Ci] -> [B, Ci*Ti, 1]
        x = x.view(x.shape[0], -1, 1)
        fmap.append(x)# .append([B, Ci*Ti])
        
        return tuple(fmap)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, input_dim, cond_dim=0, init_channels=32, periods=None, kernel_size=5, stride=3):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods if periods is not None else [2, 3, 5, 7, 11]
        disp_kwargs = {'init_channels': init_channels, 'input_dim': input_dim, 'kernel_size': kernel_size, 'stride': stride, 'cond_dim': cond_dim}
        self.discriminators = nn.ModuleList()
        for period in self.periods:
            self.discriminators.append(DiscriminatorP(period, **disp_kwargs))
    
    def main(self, y, cond=None):# [B, T, C]
        y_d = []
        fmap = []
        for i, d in enumerate(self.discriminators):
            fmap_t = d(y, cond=cond)
            y_d_t = fmap_t[-1]
            y_d.append(y_d_t)
            fmap.extend(fmap_t)
        return y_d, fmap
    
    def forward(self, y, y_hat, cond=None):
        y_d_rs, fmap_rs = self.main(y    , cond=cond)
        y_d_gs, fmap_gs = self.main(y_hat, cond=cond)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False, input_dim=1, cond_dim=0, init_channels=128, partial_padding=True):
        super(DiscriminatorS, self).__init__()
        self.dwt = None
        
        assert init_channels%16==0, f'init_channels must be multiple of 16, got {init_channels}'
        self.strides = [1, 2, 2, 4, 4, 1, 1]
        conv_kwargs = {'partial_padding': partial_padding, 'spectral_norm': use_spectral_norm, 'weight_norm': not use_spectral_norm, 'act_func': nn.LeakyReLU(D_LRELU_SLOPE), 'cond_dim': cond_dim}
        self.convs = nn.ModuleList([
            CondConv(      input_dim, 1*init_channels, kernel_size=15, stride=self.strides[0], **conv_kwargs,          ), # 1
            CondConv(1*init_channels, 1*init_channels, kernel_size=41, stride=self.strides[1], **conv_kwargs, groups= 4), # 2
            CondConv(1*init_channels, 2*init_channels, kernel_size=41, stride=self.strides[2], **conv_kwargs, groups=16), # 3
            CondConv(2*init_channels, 4*init_channels, kernel_size=41, stride=self.strides[3], **conv_kwargs, groups=16), # 4
            CondConv(4*init_channels, 8*init_channels, kernel_size=41, stride=self.strides[4], **conv_kwargs, groups=16), # 5
            CondConv(8*init_channels, 8*init_channels, kernel_size=41, stride=self.strides[5], **conv_kwargs, groups=16), # 6
            CondConv(8*init_channels, 8*init_channels, kernel_size= 5, stride=self.strides[6], **conv_kwargs,          ), # 7
        ])
        self.conv_post = CondConv(8*init_channels, 1, kernel_size=3, **{**conv_kwargs, 'act_func': None})
    
    def forward(self, x: torch.Tensor, cond=None):# [B, T, C]
        assert x.dim() == 3, f'got {x.dim()} dims, expected 3. x.shape = {x.shape}'
        fmap = []
        for i, l in enumerate(self.convs):
            x: torch.Tensor = l(x, cond=cond)
            fmap.append(x)
        x = self.conv_post(x, cond=cond)
        fmap.append(x)
        
        return tuple(fmap)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_dim, n_blocks, cond_dim=0, init_channels=128):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=bool(i==0), input_dim=input_dim, cond_dim=cond_dim, init_channels=init_channels) for i in range(n_blocks)
        ])
    
    def main(self, y, cond=None):# [B, T, C]
        y_d = []
        fmap = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = avg_pool1d(y, 2)
            fmap_r = d(y, cond=cond)
            y_d_r = fmap_r[-1]
            y_d.append(y_d_r)
            fmap.extend(fmap_r)
        return y_d, fmap
    
    def forward(self, y, y_hat, cond=None):
        y_d_rs, fmap_rs = self.main(y    , cond=cond)
        y_d_gs, fmap_gs = self.main(y_hat, cond=cond)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SFDiscriminator(nnModule):
    def __init__(self, start_c, end_c, window_size, conv_params, conv_params_list):
        super().__init__()
        self.start_c = start_c
        self.end_c = end_c
        self.window_size = window_size
        self.input_c = end_c - start_c
        
        in_dim = 1
        strides = []
        self.convs = nn.ModuleList()
        for conv_params_local in conv_params_list:
            conv_params_local = {**conv_params, **conv_params_local}
            self.convs.append(
                CondConv2d(in_dim, **conv_params_local)
            )
            in_dim = conv_params_local['out_channels']
            stride = conv_params_local.get('stride', [0, 1])
            stride = stride[1] if isinstance(stride, (list, tuple)) else (stride if type(stride) is int else 1) 
            strides.append(stride)
        
        self.post = ConvNorm(conv_params_list[-1]['out_channels']*(self.input_c//prod(strides)), 1)
    
    def window(self, tensors, lens):
        if self.window_size is None:
            return tuple(tensors), lens
        
        start_indexes = []
        for i, x_item in enumerate(tensors[0].unbind(0)):
            T_item = lens[i].item()
            max_index = max(T_item - self.window_size, 0)
            start_index = random.randint(0, max_index) if max_index > 0 else 0
            start_indexes.append(start_index)
        
        tensors_out = []
        for tensor in tensors:
            if tensor.shape[1] != 1:
                tensors_out.append(
                    torch.cat([Fpad(t.unsqueeze(0), (0, self.window_size))[:, start_indexes[i]: start_indexes[i]+self.window_size] for i, t in enumerate(tensor.unbind(0))], dim=0).to(tensor)
                )
            else:
                tensors_out.append(tensor)
        
        lens = torch.stack([l.sub(start_indexes[i]).clamp(max=self.window_size) for i, l in enumerate(lens.unbind(0))], dim=0)
        return tuple(tensors_out), lens
    
    def forward(self, x, cond=None, mask=None, lens=None):# [B, T, C], [B, T, C], [B, T, 1], [B, 1, 1]
        # window and slice
        (x, cond, mask), lens = self.window([x, cond, mask], lens=lens)# [B, window_size, C], [B, window_size, C], [B, window_size, C]
        x = x[:, :, self.start_c:self.end_c] # [B, window_size, input_c]
        
        # run 2d Convs
        x = x.unsqueeze(3)# [B, window_size, input_c, 1] / [B, T1, T2, C]
        for conv in self.convs:
            if conv.stride[0] != 1:
                cond = cond[:, ::conv.stride[0]]# -> [B, T1//stride, T2, C]
                mask = mask[:, ::conv.stride[0]]# -> [B, T1//stride, T2, 1]
            x = conv(x, cond=cond.unsqueeze(2), mask=mask)
        B, T1, T2, C = x.shape
        x = x.reshape(B, T1, T2*C)
        
        # run post and return output
        x = self.post(x)
        if mask is not None:
            x.masked_fill_(~mask, 0.0)
        return x, lens

# inspired by https://arxiv.org/pdf/2009.01776.pdf
# [not paper accurate]
class GroupedDiscriminator1d(nnModule):
    def __init__(self, channels, input_dim, window_sizes, conv_params, conv_params_list):
        super().__init__()
        assert len(window_sizes) == len(channels), f'got different lengths for window_sizes and channels. Expected {len(window_sizes)} and {len(channels)} to match.'
        assert all(sc >= 0 for sc, ec in channels), f'got start_channel less than zero. expected 0 or more.'
        assert all(ec <= input_dim for sc, ec in channels), f'got end_channel > input_dim. expected {input_dim} or lower.'
        self.window_sizes = window_sizes # [window_size1, window_size2, window_size3, ...]
        self.channels = channels  # [[start_channel, end_channel], [start_channel, end_channel], ...]
        self.n_sfs = len(channels)  # number of discriminators
        
        self.sf_discriminators = nn.ModuleList()
        for i in range(self.n_sfs):
            start_c, end_c = self.channels[i]
            window_size = self.window_sizes[i]
            self.sf_discriminators.append(
                SFDiscriminator(start_c, end_c, window_size, conv_params, conv_params_list)
            )
    
    def main(self, x, cond=None, lens=None): # [B, T, n_stft]
        assert lens is not None
        mask = get_mask1d(lens)
        
        x_list = []
        lens_list = []
        for sfd in self.sf_discriminators:
            x_, out_lens = sfd(x, cond, mask, lens)
            x_list.append(x_)
            lens_list.append(out_lens)
        return x_list, lens_list
    

class ResBlock1(nnModule):
    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5), n_layers=2, LRELU_SLOPE=0.1, conv_kwargs=None):
        super().__init__()
        conv_kwargs = {} if conv_kwargs is None else conv_kwargs
        if 'weight_norm' not in conv_kwargs:
            conv_kwargs['weight_norm'] = True
        self.LRELU_SLOPE = LRELU_SLOPE
        self.convs = \
            nn.ModuleList([
                nn.ModuleList([
                    CondConv(channels, channels, kernel_size, dilation=dilation if i==0 else 1, act_func=nn.LeakyReLU(LRELU_SLOPE), **conv_kwargs)
                for i in range(n_layers)])
            for dilation in dilations])
    
    def forward(self, x, cond=None):
        for convs in self.convs:
            x_res = x
            for conv in convs:
                x = conv(x, cond=cond)
            x = x + x_res
        return x/len(self.convs)


def maybe_mask(x, mask, mask_val=0.0):
    if mask is None:
        return x
    return x.masked_fill(~mask, mask_val)

class HiFiGenerator(nnModule):
    def __init__(self, h, mh, input_dim=None, cond_dim=0):
        super().__init__()
        self.hop_len = h['stft_config']['hop_len']
        self.n_upsblocks = len(mh.blocks)
        self.causal = mh.get('causal', False)
        
        out_dim = mh.initial_channels
        self.pre = ConvNorm(input_dim or h['stft_config']['n_mel'], out_dim)
        
        # upsample cond to wav_T
        self.ups_upsample_rates = [x['ups_upsample_rate'] for x in mh.blocks]
        self.multi_outputs = getattr(mh, 'multi_outputs', False)
        self.ups = nn.ModuleList()
        self.ups_resblocks = nn.ModuleList()
        self.posts = None
        if self.multi_outputs:
            self.posts = nn.ModuleList()
        for i, ups_conf in enumerate(mh.blocks):
            in_dim = out_dim
            out_dim = mh.initial_channels//(2**(i+1))
            self.ups.append(
                ConvTranspose(in_dim, out_dim,
                              ups_conf['ups_kernel_size'],
                              ups_conf['ups_upsample_rate'],
                              padding=(ups_conf['ups_kernel_size']-ups_conf['ups_upsample_rate'])//2,
                              causal=False)
            )
            self.ups_resblocks.append(nn.ModuleList([
                ResBlock1(
                    out_dim,
                    kernel_size=block_conf['kernel_size'],
                    dilations  =block_conf['dilation'],
                    n_layers   =block_conf['n_layers'],
                    LRELU_SLOPE=0.1,
                    conv_kwargs={'causal': self.causal, 'cond_dim': cond_dim},
                ) for j, block_conf in enumerate(ups_conf['resblocks'])
            ]))
            if self.multi_outputs:
                self.posts.append(
                    ConvNorm(out_dim, 1, kernel_size=3, causal=self.causal)
                )
        self.post = None
        if not self.multi_outputs:
            self.post = ConvNorm(out_dim, 1, kernel_size=7, causal=self.causal)
    
    def forward(self, gt_mel, cond=None, lens=None, mask=None):
        remaining_upsample_factor = self.hop_len
        if mask is None and lens is not None:
            mask = get_mask1d(lens)
        pr_wav = None
        x =  maybe_mask(self.pre(gt_mel/4.5), mask)
        for i in range(self.n_upsblocks):
            x = self.ups[i](x)
            x_skip = x
            for resblock in self.ups_resblocks[i]:
                x = maybe_mask(resblock(x, cond=cond), mask)
                x_skip = x_skip + x
            x = x_skip / sqrt(len(self.ups_resblocks[i])+1)
            
            remaining_upsample_factor = remaining_upsample_factor//self.ups_upsample_rates[i]
            if self.multi_outputs:
                wav_lat = self.posts[i](x).div(len(self.ups)).tanh().repeat_interleave(remaining_upsample_factor, dim=1) 
                pr_wav = wav_lat if i==0 else pr_wav + wav_lat
        if self.multi_outputs:
            pr_wav.data.detach().clamp_(min=-1.0, max=1.0)
        if not self.multi_outputs:
            pr_wav = self.post(x).tanh()
        return pr_wav

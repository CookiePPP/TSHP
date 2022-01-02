
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from numpy import prod

from TSHP.utils.misc_utils import zip_equal

from TSHP.utils.modules.utils import get_mask1d, maybe_cat, Fpad

from torch import Tensor
from typing import List, Tuple, Optional

from TSHP.utils.modules.core import nnModule, ConvNorm, reparameterize, ResBlock
from TSHP.utils.modules.loss_func.common import kld_loss
from TSHP.utils.modules.rnn import LSTMCellWithZoneout

class TopDownBlock(nnModule):
    def __init__(self, scale, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet: bool, is_top_block: bool, normal_weight: float = 0.05,
                 global_tokens: bool = False, kernel_size=None, cond_dim=0, act_func='selu', rezero=True, batch_norm=True, conv_kwargs=None):
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        if kernel_size is None:
            kernel_size = [1, 3, 3, 1]
        self.scale = scale
        self.hdn_dim    = hdn_dim
        self.btl_dim    = btl_dim
        self.cond_dim   = cond_dim
        self.latent_dim = latent_dim
        self.global_tokens = global_tokens
        
        self.rezero = rezero
        if self.rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)
        not_top = not is_top_block
        
        self.enc   = ResBlock(      2*btl_dim+cond_dim, 2*latent_dim        , hdn_dim, kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, n_blocks=1, n_layers=n_layers, w_gain=0.4**(1/n_layers), skip_all_res=True, rezero=False, batch_norm=batch_norm, **conv_kwargs)
        self.prior = ResBlock(not_top*btl_dim+cond_dim, 2*latent_dim+btl_dim, hdn_dim, kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, n_blocks=1, n_layers=n_layers, w_gain=0.4**(1/n_layers), skip_all_res=True, rezero=False, batch_norm=batch_norm, **conv_kwargs)
        
        self.zscale = (1/latent_dim) * 0.1
        self.normal_weight = normal_weight
        self.zproj = ResBlock(latent_dim, btl_dim, hdn_dim, cond_dim=cond_dim, act_func=act_func, n_blocks=0, n_layers=0, skip_all_res=True, rezero=False, batch_norm=batch_norm, **conv_kwargs)
        if use_post_resnet:
            self.resnet = ResBlock(btl_dim, btl_dim, hdn_dim, kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, n_blocks=int(use_post_resnet), n_layers=n_layers, w_gain=0.4**(1/n_layers), rezero=rezero, batch_norm=batch_norm, **conv_kwargs)
    
    def get_gt_z(self, x, cond, mask, z_acts):
        z_mu, z_logvar = self.enc(
            maybe_cat((x, z_acts, cond), dim=2),
            cond, mask,
        ).chunk(2, dim=2)
        z_mu     = z_mu     * self.zscale
        z_logvar = z_logvar * self.zscale
       #print(x.std().item(), z_acts.std().item(), cond.std().item())
       #print(z_mu.max().item(), z_mu.std().item(), z_logvar.max().item(), z_logvar.std().item())
        if self.global_tokens:
            z_mu     = z_mu    .mean(1, keepdim=True)
            z_logvar = z_logvar.mean(1, keepdim=True)
        return z_mu, z_logvar
    
    def get_pr_z(self, x_res, cond, mask):
        zp_mu, zp_logvar, x = self.prior(
            maybe_cat((x_res, cond), dim=2),
            cond, mask,
        ).split([self.latent_dim, self.latent_dim, self.btl_dim], dim=2)
        zp_mu     = zp_mu     * self.zscale
        zp_logvar = zp_logvar * self.zscale
        if self.rezero:
            x = x * self.res_weight
        if x_res is not None:
            x = x + x_res
        if self.global_tokens:
            zp_mu     = zp_mu    .mean(1, keepdim=True)
            zp_logvar = zp_logvar.mean(1, keepdim=True)
        return zp_mu, zp_logvar, x
    
    def post(self, z_mu, z_logvar, x, cond, mask):
        if z_mu is not None and not z_mu.isfinite().all():
            print('got non-finite elements in z_mu')
        if z_logvar is not None and not z_logvar.isfinite().all():
            print('got non-finite elements in z_logvar')
        z = reparameterize(z_mu, z_logvar, self.inferencing if self.inferencing is not False else self.training)
        if self.global_tokens:
            z = z.repeat(1, cond.shape[1], 1)
        if z is not None and not z.isfinite().all():
            print('got non-finite elements in z')
        z_embed = self.zproj(z, cond, mask)
        if z_embed is not None and not z_embed.isfinite().all():
            print('got non-finite elements in self.zproj output')
        x = x + z_embed
        if hasattr(self, 'resnet'):
            x = self.resnet(x, cond, mask)
        if x is not None and not x.isfinite().all():
            print('got non-finite elements in self.resnet output')
        return x
    
    def check_shapes(self, x, z_acts, cond, mask):
        if x is not None:
            if z_acts is not None:
                assert x.shape == z_acts.shape, f'got shapes {x.shape} and {z_acts.shape}'
            if cond is not None:
                assert x.shape[1] == cond.shape[1], f'got lengths {x.shape[1]} and {cond.shape[1]}, both should match'
            if mask is not None:
                assert x.shape[1] == mask.shape[1], f'got lengths {x.shape[1]} and {mask.shape[1]}, both should match'
    
    def infer(self, x, cond, mask):
        self.check_shapes(x, None, cond, mask)
        zp_mu, zp_logvar, x = self.get_pr_z(x, cond, mask)
        return self.post(zp_mu, zp_logvar, x, cond, mask)
    
    def forward(self, x, z_acts, cond, mask):# [B, T, C], ..., [B, T, 1]
        self.check_shapes(x, z_acts, cond, mask)
        if x is not None and not x.isfinite().all():
            print('got non-finite elements in input')
        
        zp_mu, zp_logvar, x = self.get_pr_z(x, cond, mask)
        z_mu ,  z_logvar    = self.get_gt_z(x, cond, mask, z_acts)
        kld = kld_loss(z_mu, z_logvar, zp_mu, zp_logvar, normal_weight=self.normal_weight)# [B, T, 1]
        
        if x is not None and not x.isfinite().all():
            print('got non-finite elements in self.enc output')
        return self.post(z_mu, z_logvar, x, cond, mask), kld# [B, T, C], [B, T, 1]
 
class DecBlock(nnModule):# N*ResBlock + Downsample + N*TopDownBlock + Upsample + Conv1d Grouped Cond
    def __init__(self, scale, n_blocks, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, is_last_scale, kernel_size=None, cond_dim=0, act_func='selu', rezero=True, batch_norm=False, conv_kwargs=None):
        super().__init__()
        self.encblocks = nn.ModuleList()
        self.decblocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            is_top_block = is_last_scale and bool(block_idx==0)
            self.encblocks.append(ResBlock(
                btl_dim+cond_dim, btl_dim, hdn_dim, n_blocks=1, n_layers=n_layers,
                kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, **conv_kwargs, w_gain=1.55**(1/n_layers)))
            self.decblocks.append(TopDownBlock(
                scale, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, is_top_block,
                kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, conv_kwargs=conv_kwargs))
        
        self.scale = scale
    
    def downsample(self, x):
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=self.scale).transpose(1, 2)
    
    def upsample(self, x):
        return F.interpolate(x.transpose(1, 2), scale_factor=self.scale).transpose(1, 2)
    
    def encode(self, z_acts, cond, mask):
        for i, encblock in enumerate(self.encblocks):
            z_acts = encblock(maybe_cat(z_acts, cond), cond, mask)
        return z_acts
    
    def decode(self, x, z_acts, cond, mask):
        kld_list = []
        for i, decblock in enumerate(self.decblocks):
            x, kld = decblock(x, z_acts, cond, mask)# [B, T, C], [B, T, 1]
            kld_list.append(kld.masked_fill(~mask, 0.0))
        return x, kld_list
    
    def infer(self, x, cond, mask):
        for decblock in self.decblocks:
            x = decblock.infer(x, cond, mask)# [B, T, C], [B, T, 1]
        return x


class LSTMGlobalDecBlock(nnModule):# N*ResBlock + Global Downsample + N*TopDownBlock + Upsample to cond + Conv1d Grouped Cond
    def __init__(self, n_blocks, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, cond_dim=0, act_func='selu', rezero=True, batch_norm=False, conv_kwargs=None):
        super().__init__()
        self.encblocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            self.encblocks.append(ResBlock(
                btl_dim+cond_dim, btl_dim, hdn_dim, n_blocks=1, n_layers=n_layers,
                kernel_size=1, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, **conv_kwargs, w_gain=1.55**(1/n_layers)))
        
        self.decblocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            is_top_block = bool(block_idx==0)
            self.decblocks.append(TopDownBlock(
                'global', hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, is_top_block,
                kernel_size=1, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, conv_kwargs=conv_kwargs))
    
    def encode(self, z_acts, cond, mask):# [B, T, btl_dim], [B, T, cond_dim], [B, T, 1]
        for i, encblock in enumerate(self.encblocks):
            z_acts = encblock(maybe_cat(z_acts, cond), cond, mask)
        return z_acts# [B, T, btl_dim]
    
    def decode(self, x, z_acts, cond, mask):# [B, T, btl_dim], [B, T, btl_dim], [B, T, cond_dim], [B, T, 1]
        kld_list = []
        for i, decblock in enumerate(self.decblocks):
            x, kld = decblock(x, z_acts, cond, mask)# [B, T, btl_dim], [B, T, btl_dim], [B, T, cond_dim], [B, T, 1]
            kld_list.append(kld.masked_fill(~mask, 0.0))
        return x, kld_list
    
    def infer(self, x, cond, mask):
        for decblock in self.decblocks:
            x = decblock.infer(x, cond, mask)# [B, T, C], [B, T, 1]
        return x


class ConvGlobalDecBlock(nnModule):# N*ResBlock + Global Downsample + N*TopDownBlock + Upsample to cond + Conv1d Grouped Cond
    def __init__(self, n_blocks, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, cond_dim=0, act_func='selu', rezero=True, batch_norm=False, conv_kwargs=None):
        super().__init__()
        self.encblocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            self.encblocks.append(ResBlock(
                btl_dim+cond_dim, btl_dim, hdn_dim, n_blocks=1, n_layers=n_layers,
                kernel_size=1, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, **conv_kwargs, w_gain=1.55**(1/n_layers)))
        
        self.decblocks = nn.ModuleList()
        for block_idx in range(n_blocks):
            is_top_block = bool(block_idx==0)
            self.decblocks.append(TopDownBlock(
                'global', hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, is_top_block, global_tokens=True,
                kernel_size=1, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, conv_kwargs=conv_kwargs))
    
    def encode(self, z_acts, cond, mask):
        for i, encblock in enumerate(self.encblocks):
            z_acts = encblock(maybe_cat(z_acts, cond), cond, mask)
        return z_acts
    
    def decode(self, x, z_acts, cond, mask):
        kld_list = []
        for i, decblock in enumerate(self.decblocks):
            x, kld = decblock(x, z_acts, cond, mask)# [B, T, C], [B, T, 1]
            kld_list.append(kld.masked_fill(~mask, 0.0))
        return x, kld_list
    
    def infer(self, x, cond, mask):
        for decblock in self.decblocks:
            x = decblock.infer(x, cond, mask)# [B, T, C], [B, T, 1]
        return x


class VDRESVAE(nnModule):
    def __init__(self, in_dim: int, cond_dim: int, scales: List[int], scales_n_blocks: List[int], hdn_dim: int, btl_dim: int, latent_dim: int,
                 n_layers: int, use_post_resnet: bool, cond_prenet_n_blocks: int = 0, cond_prenet_kwargs=None, use_global_scale: bool = False, global_scale_n_blocks: int = 0, rnn_global_scale: bool = True,
                 out_dim: int = None, kernel_size=None, act_func='selu', rezero=True, batch_norm=False, **conv_kwargs):
        super().__init__()
        if cond_prenet_kwargs is None:
            cond_prenet_kwargs = {}
        if kernel_size is None:
            kernel_size = [1, 3, 3, 1]
        if out_dim is None:
            out_dim = in_dim
        self.total_scale = prod(scales)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cond_dim = cond_dim
        
        if cond_prenet_n_blocks:
            self.cond_prenet = ResBlock(cond_dim, cond_dim, max(cond_dim, hdn_dim),
                                        n_blocks=cond_prenet_n_blocks, **cond_prenet_kwargs)
        
        if in_dim != btl_dim:
            self.pre = ConvNorm(in_dim, btl_dim)
        
        self.blocks = nn.ModuleList()
        self.scales = scales
        self.scale_n_blocks = scales_n_blocks # list[int] <- n_blocks in each scale
        self.n_scales = len(scales_n_blocks)
        for i, (n_blocks, scale) in enumerate(zip_equal(scales_n_blocks, scales)):
            is_last_scale = bool(i+1==len(scales_n_blocks)) and not use_global_scale
            self.blocks.append(
                DecBlock(scale, n_blocks, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet, is_last_scale,
                         kernel_size=kernel_size, cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, conv_kwargs=conv_kwargs)
            )
       
        self.use_global_scale = use_global_scale
        if use_global_scale:
            assert global_scale_n_blocks > 0, "global_scale_n_blocks must be greater than 0 if use_global_scale"
            self.global_scale = ConvGlobalDecBlock(
                global_scale_n_blocks, hdn_dim, btl_dim, latent_dim, n_layers, use_post_resnet,
                cond_dim=cond_dim, act_func=act_func, rezero=rezero, batch_norm=batch_norm, conv_kwargs=conv_kwargs)
        
        if btl_dim != out_dim:
            self.post = ConvNorm(btl_dim, out_dim)
    
    def downsample_to_list(self, x):
        if x.shape[1] != 1:
            x = Fpad(x, (0, (-x.shape[1]) % self.total_scale))# pad to multiple of max downscale
        x_list = []
        for i, block in enumerate(self.blocks):
            x_list.append(x)
            if x.shape[1] != 1:
                x = block.downsample(x)
        
        return x_list
    
    def get_mask_list(self, lens=None, mask=None, type='floor'):
        if mask is None:
            mask = get_mask1d(lens)# [B, mel_T, 1]
        masks = self.downsample_to_list(mask.float())
        if   type == 'floor':
            masks = [mask.floor().bool() for mask in masks]
        elif type == 'round':
            masks = [mask.round().bool() for mask in masks]
        elif type == 'ceil' :
            masks = [mask. ceil().bool() for mask in masks]
        else:
            raise NotImplementedError
        return masks
    
    def encode(self, c_list, m_list, x):
        z_acts_list = [x, ]
        for i in range(len(self.blocks)):# [0, ..., n_blocks]
           #print(i, x.shape, c_list[i].shape, m_list[i].shape)
            is_top_block = i+1 == len(self.blocks)
            x = self.blocks[i].encode(x, c_list[i], m_list[i])# -> [B, T, C]
            
            if not is_top_block:
                x = self.blocks[i].downsample(x)# -> [B, T//scale, C]
            z_acts_list.append(x)
        
        if self.use_global_scale:
           #print('g', x.shape, c_list[-1].shape, m_list[-1].shape)
            x = self.global_scale.encode(x, c_list[-1], m_list[-1])
            z_acts_list.append(x)
        
        return z_acts_list
    
    def decode(self, c_list, m_list, z_acts):
        kld_list = []
        
        x = None
        
        if self.use_global_scale:
           #print('g', x, z_acts[-1].shape, c_list[-1].shape, m_list[-1].shape)
            x, kld = self.global_scale.decode(x, z_acts[-1], c_list[-1], m_list[-1])
            kld_list.extend(kld)
        
        for i in range(len(self.blocks))[::-1]:# [n_blocks, ..., 0]
            is_top_block = i+1 == len(self.blocks)
            if not is_top_block:
                x = self.blocks[i].upsample(x)
           #print(i, x.shape, z_acts[i].shape, c_list[i].shape, m_list[i].shape)
            x, kld = self.blocks[i].decode(x, z_acts[i], c_list[i], m_list[i])
            kld_list.extend(kld)
        
        return x, kld_list
    
    def decode_infer(self, c_list, m_list):
        x = None
        
        if self.use_global_scale:
            x = self.global_scale.infer(x, c_list[-1], m_list[-1])
        
        for i in range(len(self.blocks))[::-1]:
            is_top_block = i+1 == len(self.blocks)
            if not is_top_block:
                x = self.blocks[i].upsample(x)
            x = self.blocks[i].infer(x, c_list[i], m_list[i])
        return x
    
    def kld_loss(self, kld_list, m_list):
        kld_total = sum(t.sum([1, 2], dtype=torch.float) for t in kld_list)# -> [B]
        len_total = sum(t.sum([1, 2], dtype=torch.float)*self.scale_n_blocks[i] for i, t in enumerate(m_list[1:]))# -> [B]
        return kld_total / (len_total + int(self.use_global_scale))# [B]
    
    def forward(self, x_in, cond, lens=None, mask=None):# [B, T, C], [B, T, C], [B, ...]
        if hasattr(self, 'pre'):
            x_in = self.pre(x_in)
        xi = Fpad(x_in, (0, (-x_in.shape[1]) % self.total_scale))# pad to multiple of max downscale
        assert xi.isfinite().all(), 'non-finite elements found on VDRESVAE inputs'
        
        if getattr(self, 'cond_prenet', None):
            cond = self.cond_prenet(cond)
        
        c_list = self.downsample_to_list(cond)# -> list
        m_list = self.get_mask_list(lens=lens, mask=mask)# -> list
        
        z_acts_list = self.encode(c_list, m_list, xi)
       #print("z_acts_list: ", [z.std().item() for z in z_acts_list])
       #print("c_list: ", [z.std().item() for z in c_list])
        x, kld_list = self.decode(c_list, m_list, z_acts_list)
        kld_total = self.kld_loss(kld_list, m_list)# [B]
        
        assert x.shape[1] == xi.shape[1], f'got output length of {x.shape[1]}, expected {xi.shape[1]}'
        x = x[:, :x_in.shape[1], :]
        if hasattr(self, 'post'):
            x = self.post(x)
        return x, kld_total# [B, T, out_dim], [B]
    
    def infer(self, cond, lens=None, mask=None):# [B, T, C], [B, T, C], [B, ...]
        if getattr(self, 'cond_prenet', None):
            cond = self.cond_prenet(cond)
        
        c_list = self.downsample_to_list(cond)# -> list
        m_list = self.get_mask_list(lens=lens, mask=mask)# -> list
        
        x = self.decode_infer(c_list, m_list)
        
        assert x.shape[1] == c_list[0].shape[1], f'got output length of {x.shape[1]}, expected {c_list[0].shape[1]}'
        x = x[:, :cond.shape[1], :]
        if hasattr(self, 'post'):
            x = self.post(x)
        return x # [B, T, out_dim]

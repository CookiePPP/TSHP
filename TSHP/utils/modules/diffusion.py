import math
import os
import json
from typing import Tuple

import torch
from TSHP.utils.modules.loss_func.common import get_mean_errors
from TSHP.utils.modules.utils import get_mask1d
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from functools import partial
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)# get select a indexes along last dim using t
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, n_channels, lin_start: float=1e-4, lin_end: float=0.06, lin_n_steps: int=100, logspace: bool=False, predict_noise=True):
        super().__init__()
        self.n_channels = n_channels
        self.set_noise_schedule(lin_start, lin_end, lin_n_steps, logspace)
        self.predict_noise = predict_noise
    
    def set_noise_schedule(self, lin_start, lin_end, lin_n_steps, logspace: bool=False, device='cpu'):
        if logspace:
            betas = np.exp(np.linspace(np.log(lin_start), np.log(lin_end), lin_n_steps, endpoint=True))# T == len(schedule_list)
        else:
            betas = np.linspace(lin_start, lin_end, lin_n_steps, endpoint=True)# T == len(schedule_list)
                                                     # [0.0001, ..., 0.0301, ..., 0.0600]
        
        alphas = 1. - betas# -> [0.9999, ..., 0.9699, ..., 0.9400]
        alphas_cumprod = np.cumprod(alphas, axis=0)# [0.9999, ..., <0.9699, ..., <0.9400]
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])# [1.0000, 0.9999, ..., <0.9699, ...]
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        to_torch = partial(torch.tensor, device=device, dtype=torch.float32)# ??? numpy/list -> pytorch?
        
        self.register_buffer('betas', to_torch(betas), persistent=False)
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod), persistent=False)
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev), persistent=False)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)), persistent=False)                 # sqrt(a_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)), persistent=False)  # sqrt(1.0 - a_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)), persistent=False)      #  log(1.0 / a_cumprod - 1)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)), persistent=False)# sqrt(1.0 - a_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20*self.num_timesteps))), persistent=False)
        
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)), persistent=False)
        
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)), persistent=False)
    
    def get_noise_level(self, gt_X, random_nl=True):
        B = gt_X.shape[0]
        if random_nl:
            t = torch.randint(0, self.num_timesteps, (B,), device=gt_X.device)
        else:
            if not self.training:  # randomly sample noise
                t = F.interpolate(torch.arange(0, self.num_timesteps, device=gt_X.device).view(1, 1, -1).float(), size=B, mode='linear',
                                  align_corners=True).round().long().view(-1)[torch.randperm(B)]
                # [B] get random noise level for each item in batch
            else:  # systematic sample noise
                if not hasattr(self, 'noise_t_steps'):
                    self.noise_t_steps = torch.arange(0, self.num_timesteps, device=gt_X.device)
                    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...]
                t = self.noise_t_steps.repeat_interleave(math.ceil(B / self.num_timesteps))[:B]  # [B+?]
                
                # s.t = [0, 1, 2, 3, 4]
                # t   = [0, 1, 2, 3, 4, 0, 1]
                # s.t = [2, 3, 4, 0, 1] # roll back by len(t) % len(s.t) so next t uses next elements in order
                self.noise_t_steps = self.noise_t_steps.roll(-(t.shape[0] % self.num_timesteps), dims=0)
        noise_scalar = self.get_noise_scalar(gt_X, t)  # [B, 1, 1] get noise scalar
        return noise_scalar, t
    
    def add_noise_to_gt(self, gt_X, x_lens=None, random_nl=True):
        noise_scalar, t = self.get_noise_level(gt_X, random_nl=random_nl)
        
        gt_X_noise = torch.randn_like(gt_X) # get randn_noise
        gt_X_noisy_t     = self.q_sample(x_start=gt_X, noise=gt_X_noise, t=t)
        gt_X_noisy_tsub1 = self.q_sample(x_start=gt_X, noise=gt_X_noise, t=t.clamp(min=1)-1)
        
        # if t == 0: noisy_tsub1 = gt_X
        if t.eq(0).any():
            gt_X_noisy_tsub1 = gt_X.where(t.view(-1, 1, 1).eq(0), gt_X_noisy_tsub1)
        
        if x_lens is not None:
            x_mask = get_mask1d(x_lens)
            gt_X_noise       = gt_X_noise      .masked_fill_(~x_mask, 0.0)
            gt_X_noisy_t     = gt_X_noisy_t    .masked_fill_(~x_mask, 0.0)
            gt_X_noisy_tsub1 = gt_X_noisy_tsub1.masked_fill_(~x_mask, 0.0)
        return (noise_scalar, t), (gt_X_noise, gt_X_noisy_t, gt_X_noisy_tsub1)
    
    def predict_start_from_noise(self, x_noisy, t, x_noise) -> torch.Tensor:
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_noisy.shape) * x_noisy -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_noise.shape) * x_noise
        )
    
    def q_posterior(self,
                    x,      # pr_mel
                    x_noisy,# loop_mel
                    t) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:     # noise_step
        """Blend pr_mel with loop_mel (to make small steps across the probability distribution)."""
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x.shape) * x +
                extract(self.posterior_mean_coef2, t, x_noisy.shape) * x_noisy
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x.shape)
        posterior_std_clipped = posterior_log_variance_clipped.mul(0.5).exp()
        return posterior_mean, posterior_std_clipped, posterior_log_variance_clipped# ???, ???, ???
    
    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
        if type(t) is int:
            t: torch.Tensor = torch.tensor([t, ], device=x_start.device, dtype=torch.long).expand(x_start.shape[0])
        # adds t scale noise to x_start. Called on the input before every denoiser pass
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def get_noise_scalar(self, x, t) -> torch.Tensor:
        return extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
    
    def get_noise_scalars(self, x, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t_tensor: torch.Tensor = torch.tensor([t, ], device=x.device, dtype=torch.long).expand(x.shape[0])
        noise_scalar = self.get_noise_scalar(x, t_tensor)
        return noise_scalar, t_tensor
    
    def get_mean_errors(self, gt_X, gt_X_noise, pr_X, pr_X_noise, lens=None, mask=None):
        if self.n_channels is not None:
            assert self.n_channels == gt_X.shape[2]
            assert self.n_channels == pr_X.shape[2]
            assert self.n_channels == gt_X_noise.shape[2]
        if self.predict_noise:
            return get_mean_errors(gt_X_noise, pr_X_noise, lens=lens, mask=mask)
        else:
            return get_mean_errors(gt_X, pr_X, lens=lens, mask=mask)
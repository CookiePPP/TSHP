import torch
import torch.nn as nn
from TSHP.utils.modules.core import nnModule

from typing import Optional
from torch import Tensor

from TSHP.utils.modules.utils import get_mask1d


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-05,
                 momentum: Optional[float] = 0.1,
                 eval_only_momentum: bool = False,
                 logify_inputs: bool = False,
                 log_clamp_val: float = -2.0,
                 mask_pad_val: float = 0.0,
                 affine: bool = True,
                 aux_weight: Optional[float] = None,
                 aux_bias: Optional[float] = None,
                 sum_unit_var: bool = False,
                 track_running_stats: bool = True
                 ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.eval_only_momentum = eval_only_momentum
        self.logify_inputs = logify_inputs
        if self.logify_inputs:
            self.log_clamp_val = log_clamp_val
        else:
            self.log_clamp_val = None
        self.mask_pad_val = mask_pad_val
        self.aux_weight = aux_weight
        self.aux_bias = aux_bias
        self.sum_unit_var = sum_unit_var
    
    def log_input(self, x, mask=None):
        if mask is None:
            return x.log().clamp(min=self.log_clamp_val)
        else:
            return torch.where(mask, x.log().clamp(min=self.log_clamp_val), x)
    
    def exp_output(self, x, mask=None):
        if mask is None:
            return x.exp()
        else:
            return torch.where(mask, x.exp(), x)
    
    def forward(self, x, mask=None, lens=None):# [B, T, C], [B, T, 1]
        if mask is None and lens is not None:
            mask = get_mask1d(lens) # [B, T, 1]
        assert mask is None or x.shape[1] == mask.shape[1], f'lens {x.shape[1]} and {mask.shape[1]} do not match' 
        dtype = next(p.dtype for p in self.buffers())
        if not x.isfinite().all():
            raise ValueError('got non-finite elements in batchnorm input x')
        if self.logify_inputs:
            x = self.log_input(x, mask)
        else:
            x = x.clone()
        
        B, T, C = x.shape
        x = x.reshape(B*T, -1)# [B*T, C]
        if B*T <= 2:
            return x.to(dtype).view(B, T, C)
        
        if not x.isfinite().all():
            raise ValueError('got non-finite elements in batchnorm input x')
        if mask is not None:
            x = self.main_masked(B, T, mask, x)
        else:
            x = self.main(x)
        if not x.isfinite().all():
            raise ValueError('got non-finite elements in batchnorm output x')
        return x.to(dtype).view(B, T, C)
    
    def super_forward(self, x):
        x = super().forward(x)
        if self.aux_weight is not None:
            x = x.mul(self.aux_weight/x.shape[1] if self.sum_unit_var else self.aux_weight)
        if self.aux_bias is not None:
            x = x.add(self.aux_bias)
        return x
    
    def main(self, x):
        if self.eval_only_momentum or not self.training:
            if x.shape[0] > 1:
                x = self.super_forward(x)  # [B*T, C]
        else:
            if x.shape[0] > 1:
                _ = self.super_forward(x)  # [B*T, C]
            
            self.train(False)
            x = self.super_forward(x)  # [B*T, C]
            self.train(True)
        return x
    
    def main_masked(self, B, T, mask, x):
        mask = mask.view(B * T)  # [B*T]
        assert x.shape[:1] == mask.shape[:1], f'mask has shape {mask.shape}, expected {x.shape}'
        if self.eval_only_momentum or not self.training:
            x[mask] = self.super_forward(x[mask])  # [B*T, C]
        else:
            _ = self.super_forward(x[mask])  # [B*T, C]
            
            self.train(False)
            x[mask] = self.super_forward(x[mask])  # [B*T, C]
            self.train(True)
        return x

    def inverse(self, x, mask=None, lens=None):# [B, T, C], [B, T, 1]
        assert self.track_running_stats, 'track_running_stats required for inverse() method'
        if mask is None and lens is not None:
            mask = get_mask1d(lens)
        B, T, C = x.shape
        
        if mask is not None:
            x_orig = x.clone()
        
        # x2 = maybe_log(x1)
        # y1 = (x-mean)/var.sqrt()
        # y2 = (y1*weight)+bias
        # y3 = (y2*weight)+bias
    
        if self.aux_bias is not None:
            x = x.sub(self.aux_bias)
        if self.aux_weight is not None:
            x = x.div(self.aux_weight/x.shape[1] if self.sum_unit_var else self.aux_weight)
    
        if self.affine:
            weight = self.weight[None, None, :] # [1, 1, C]
            bias   = self.bias  [None, None, :] # [1, 1, C]
            x = (x-bias)/weight
    
        mean = self.running_mean[None, None, :] # [1, 1, C]
        var  = self.running_var [None, None, :] # [1, 1, C]
        x = x.mul(var.sqrt()).add(mean)
    
        if self.logify_inputs:
            x = self.exp_output(x, mask)
        if mask is not None:
            x = torch.where(mask, x, x_orig)
        return x.view(B, T, C)

if __name__ == '__main__':
    for eval_only_momentum in [False,]:
        for momentum in [1.0, 0.9, 0.1]:
            for affine in [False, True]:
                batchnorm_layer = BatchNorm1d(
                    2,
                    momentum=momentum,
                    eval_only_momentum=eval_only_momentum,
                    affine=affine,
                    eps=1e-9,
                ).cuda()
                batchnorm_layer.train()
                
                x = torch.randn(1, 16, 2, device='cuda')
                
                x_orig = x.clone()
                for i in range(1000):
                    _ = batchnorm_layer(x)
                assert (x == x_orig).all(), 'check for any inplace modification'
                
                y = batchnorm_layer(x)
                
                x_ = batchnorm_layer.inverse(y)
                
                print(x.std(1).view(-1).tolist())
                print(batchnorm_layer.running_var.sqrt().tolist())
                
                assert x.mul(10).round().eq(x_.mul(10).round()).all(), f'\nmomentum={momentum}, eval_only_momentum={eval_only_momentum}, affine={affine} FAILED\n' \
                                                                       f'got x = {x .double().mul(1e5).round().div(1e5).tolist()}\n' \
                                                                       f'x_out = {x_.double().mul(1e5).round().div(1e5).tolist()}\n' \
                                                                       f'Expected x == x_out'
                print(f'momentum={momentum}, eval_only_momentum={eval_only_momentum}, affine={affine} PASSED\n\n')
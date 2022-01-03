import contextlib
import copy
import datetime
import math
import os
import time
import warnings
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from TSHP.modules.train_utils import deepto
from TSHP.utils.misc_utils import zip_equal

from TSHP.utils.saving.utils import safe_write

from TSHP.utils.load_utils import load_state_dict_force

from TSHP.utils.arg_utils import get_args
from TSHP.utils.modules.activation_funcs import get_afunc, get_afunc_gain
from torch import Tensor
from typing import List, Tuple, Optional, Union, Any

from TSHP.utils.modules.utils import get_mask1d


# example taken from https://stackoverflow.com/a/24439444 and modified
from inspect import getframeinfo, stack

def dist_barrier(n_rank=None, timeout=None, async_op=True, assert_same_call=True):
    if n_rank is None:
        n_rank = dist.get_world_size()
    if n_rank <= 1:
        return
    if timeout is None:
        timeout = datetime.timedelta(seconds=15.0)
    elif type(timeout) in [int, float]:
        timeout = datetime.timedelta(seconds=timeout)
    if assert_same_call:
        caller_loc = [None for _ in range(n_rank)]
        pos = {
            'filename': getframeinfo(stack()[1][0]).filename,
            'lineno': getframeinfo(stack()[1][0]).lineno,
        }
        dist.all_gather_object(caller_loc, pos)
        caller_loc_current = caller_loc[0]
        if not all(cl == caller_loc_current for cl in caller_loc):
            time.sleep(random.uniform(0.0, 2.0)) # add random delay so stack-traces don't perfectly overlap
            raise NotImplementedError('assert_same_call is True but dist_barrier() was called from different locations')
    
    if async_op:
        work = dist.barrier(async_op=True)
        work.wait(timeout)
    else:
        dist.barrier()

@contextlib.contextmanager
def specific_rank_first(rank_to_go_first: int, cur_rank: int, n_rank=None, timeout=None, async_op=True):
    if cur_rank != rank_to_go_first:
        dist_barrier(n_rank=n_rank, timeout=timeout, async_op=async_op, assert_same_call=False)
    yield
    if cur_rank == rank_to_go_first:
        dist_barrier(n_rank=n_rank, timeout=timeout, async_op=async_op, assert_same_call=False)

def dist_add(x, n_rank=None, deepcopy=True, timeout=None, async_op=True, assert_same_call=True):
    if n_rank is None:
        n_rank = dist.get_world_size()
    if timeout is None:
        timeout = datetime.timedelta(seconds=15.0)
    
    if n_rank == 1:
        if deepcopy:
            return copy.deepcopy(x)
        else:
            return x
    
    if assert_same_call:
        caller_loc = [None for _ in range(n_rank)]
        pos = {
            'filename': getframeinfo(stack()[1][0]).filename,
            'lineno': getframeinfo(stack()[1][0]).lineno,
        }
        dist.all_gather_object(caller_loc, pos)
        caller_loc_current = caller_loc[0]
        if not all(cl == caller_loc_current for cl in caller_loc):
            time.sleep(random.uniform(0.0, 2.0)) # add random delay so stack-traces don't perfectly overlap
            raise NotImplementedError('assert_same_call is True but dist_add() was called from different locations')
    
    if type(x) is torch.Tensor:
        xd = x.clone().to('cuda')
    else:
        xd = torch.tensor(x, device='cuda')
    
    if async_op:
        work = dist.all_reduce(xd, async_op=True)
        work.wait(timeout)
    else:
        dist.all_reduce(xd)
    
    if type(x) is torch.Tensor:
        xd = xd.to(x)
    else:
        xd = xd.item()
    return xd

def dist_mean(x, n_rank, deepcopy=True, timeout=None, async_op=True, assert_same_call=True):
    if n_rank == 1:
        return x
    if assert_same_call:
        caller_loc = [None for _ in range(n_rank)]
        pos = {
            'filename': getframeinfo(stack()[1][0]).filename,
            'lineno': getframeinfo(stack()[1][0]).lineno,
        }
        dist.all_gather_object(caller_loc, pos)
        caller_loc_current = caller_loc[0]
        if not all(cl == caller_loc_current for cl in caller_loc):
            time.sleep(random.uniform(0.0, 2.0)) # add random delay so stack-traces don't perfectly overlap
            raise NotImplementedError('assert_same_call is True but dist_mean() was called from different locations')
    x = dist_add(x, n_rank=n_rank, deepcopy=deepcopy, timeout=timeout, async_op=async_op, assert_same_call=False)
    x /= n_rank
    return x

class DictAsMember(dict):# taken from https://stackoverflow.com/a/11049286
    def __getattr__(self, name):
        try:
            value = self[name]
        except:
            raise AttributeError(f'"{name}" attribute not found.')
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value

def module_inference(module, inference):
    if hasattr(module, 'inferencing'):
        module.inference(inference)
    elif isinstance(module, nn.ModuleList):
        for module_i in module:
            module_inference(module_i, inference)

class nnModule(nn.Module):
    h: dict
    mh: dict
    def __init__(self):
        super(nnModule, self).__init__()
        self.inferencing = False
    
    def set_h(self, h):
        self.h = h
        self.mh = DictAsMember(h['model_config'])
    
    def inference(self, inference=True):
        self.inferencing = inference
        for module in self.children():
            module_inference(module, inference)
    
    def remove_norm_recursive(self, name='weight'):
        if hasattr(self, f'{name}_g') and hasattr(self, f'{name}_v'):
            nn.utils.remove_weight_norm(self, name=name)
        for module in self.children():
            if hasattr(module, 'remove_norm_recursive'):
                module.remove_norm_recursive(name)
    
    def transfer_device(self, x): # TODO: Make recursive / handle nested lists/dicts
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
        
        def update(x):
            return x if type(x) is not torch.Tensor else (
                    x.to(device, dtype) if x.dtype in [torch.float16, torch.float32, torch.float64] else x.to(device))
        
        if isinstance(x, type({})):
            return {k: update(v) for k, v in x.items()}# convert (not long) tensors to parameter dtype and move all tensors to correct device
        elif isinstance(x, (list, tuple)):
            return [update(v) for v in x]
        elif type(x) is torch.Tensor:
            return update(x)
        else:
            raise NotImplementedError(f'got type(x) = {type(x)}, expected dict, list, tuple, or Tensor')

class ModelModule(nnModule):
    iteration: torch.Tensor
    epoch: torch.Tensor
    secpr: torch.Tensor
    max_learning_rate: torch.Tensor
    best_cross_val: torch.Tensor
    best_cross_val_secpr: torch.Tensor
    lr_multiplier: torch.Tensor
    last_save_time: torch.Tensor
    
    def __init__(self):
        super(ModelModule, self).__init__()
        self.register_buffer("iteration", torch.tensor(0  , dtype=torch.long  ))
        self.register_buffer("epoch"    , torch.tensor(0.0, dtype=torch.double))
        self.register_buffer("secpr"    , torch.tensor(0.0, dtype=torch.double))
        self.register_buffer("max_learning_rate", torch.tensor(0.0, dtype=torch.double))
        # secpr when cross_val last improved
        self.register_buffer("best_cross_val"      , torch.tensor(float('inf'), dtype=torch.double))
        self.register_buffer("best_cross_val_secpr", torch.tensor(0.0, dtype=torch.double))
        # decreases during training when cross_val doesn't improve (for some extended period of time)
        self.register_buffer("lr_multiplier", torch.tensor(1.0, dtype=torch.double))
        self.register_buffer("last_save_time", torch.tensor(0.0, dtype=torch.double))
    
    def inference_mode(self):
        return self
    
    @classmethod
    def load_model(cls, checkpoint_path, train=True, h=None, reset_tracker=False, ignore_checkpoint_requirement=False):# use Model.load_model(path) to get model
        if train:
            assert h is not None, 'self.h is None, expected dict.\nConfig is required to load a model for training.'
        else:
            assert checkpoint_path or ignore_checkpoint_requirement, 'if not training, checkpoint path is expected!'
        
        changed_shape = False
        if checkpoint_path:
            d = torch.load(checkpoint_path, map_location='cpu')
            model = cls(h or d['h'])
            if 'spkrlist' in d:
                model.spkrlist = d['spkrlist']
            if 'textlist' in d:
                model.textlist = d['textlist']
            changed_shape = load_state_dict_force(model, d['state_dict'])
        else:
            model = cls(h)
        
        if reset_tracker:
            model.reset_tracker()
        
        model.train(train)
        return model, changed_shape
    
    def save_model(self, checkpoint_path: str):
        os.makedirs(os.path.split(checkpoint_path)[0], exist_ok=True)
        savedict = {}
        savedict['h'] = self.h
        savedict['state_dict'] = self.state_dict()
        if hasattr(self, 'spkrlist'):
            savedict['spkrlist'] = self.spkrlist
        if hasattr(self, 'textlist'):
            savedict['textlist'] = self.textlist
        
        # need spkrname to spkrids or spkrlist
        safe_write(savedict, checkpoint_path)
    
    def get_kwargs(self):
        self.model_args = list(get_args(self.generator.forward)) # get args for generator forward
        if hasattr(self, 'discriminator'):
            self.model_args.extend(get_args(self.discriminator.forward))  # get args for discriminator forward
            self.model_args = list(set(self.model_args))  # remove duplicate entries
        return self.model_args
    
    def reset_tracker(self):
        self.iteration.fill_(0)
        self.epoch.fill_(0.0)
        self.secpr.fill_(0.0)
        self.best_cross_val.fill_(float('inf'))
        self.best_cross_val_secpr.fill_(0.0)
        self.lr_multiplier.fill_(1.0)
        self.last_save_time.fill_(0.0)
    
    def offset_tracker(self, iteration_delta: int, epoch_delta: float, secpr_delta: float):
        self.iteration += iteration_delta
        self.epoch += epoch_delta
        self.secpr += secpr_delta
    
    def loss_stuff(self, out: dict, loss_weights: dict, weight_lens: Optional[torch.Tensor]=None):
        if weight_lens is None:
            weight_lens = out['mel_lens']
        
        # get loss value(s) for backward()
        out['loss_g'] = self.colate_losses(out['loss_dict_g'], loss_weights, weight_lens=weight_lens)
        if hasattr(self, 'discriminator'):
            out['loss_d'] = self.colate_losses(out['loss_dict_d'], loss_weights, weight_lens=weight_lens)
        
        # detach d values from compuational graph before they reach the logger
        out['d'] = {k: v.detach() for k, v in [*out.get('loss_dict_g', {}).items(), *out.get('loss_dict_d', {}).items()]}
        out['loss_dict_g'] = {k: v.detach() for k, v in out.get('loss_dict_g', {}).items()}
        if hasattr(self, 'discriminator'):
            out['loss_dict_d'] = {k: v.detach() for k, v in out.get('loss_dict_d', {}).items()}
        if out.get('d', None) is not None:
            out['loss_dict'] = self.reduce_lossdict(out['d'], self.h['n_rank'], weight_lens=weight_lens)
            out['loss_dict_path'] = self.get_path_losses(out['d'], out['loss_dict_g'], out.get('loss_dict_d', None), loss_weights, out['audiopath'], self.h['n_rank'])
        if out.get('loss_g', None) is not None:
            out['loss_g_reduced'] = dist_mean(out['loss_g'], self.h['n_rank']).item() if self.h['n_rank'] > 1 else out['loss_g'].mean().item()
        if out.get('loss_d', None) is not None:
            out['loss_d_reduced'] = dist_mean(out['loss_d'], self.h['n_rank']).item() if self.h['n_rank'] > 1 else out['loss_d'].mean().item()
        
        assert 'loss_g' in out and 'd' in out
        return out
    
    def colate_losses(self, loss_dict: dict, loss_weights: dict, weight_lens: Optional[torch.Tensor]=None, silent=False):
        # collate losses for optimizer, tracks gradients for training
        loss = None
        for k, v in loss_dict.items():
            # get loss_scale
            loss_scale = loss_weights.get(f'{k}_weight', None) or loss_weights.get(f'{k}', None)
            if loss_scale is None:
                loss_scale = getattr(self, f'{k}_weight', None)
            if loss_scale is None:
                loss_scale = 0.0
                if not silent:
                    print(f'{k} is missing loss weight')
            
            # mulitply loss by scale (and maybe weight by lengths), then add to loss output
            if loss_scale > 0.0:
                
                # reweight losses so shorter files adjust the model less than longer files
                if weight_lens is not None:
                    v = v * (weight_lens.float()/weight_lens.float().mean())
                
                # ensure v is a scalar
                if torch.is_tensor(v):
                    v = v.mean()
                elif not silent:
                    print(f'{k} is not a Tensor! got {v}')
                
                new_loss = v*loss_scale
                
                if new_loss > 40.0 or math.isnan(new_loss) or math.isinf(new_loss):
                    if not silent:
                        print(f'{k} reached {v}')
                
                if loss is not None:
                    loss = loss + new_loss
                else:
                    loss = new_loss
        return loss
    
    def reduce_lossdict(self, loss_dict: dict, world_size: int, weight_lens: Optional[torch.Tensor]=None):
        # reduce loss across all GPUs, used for logging, does not transfer gradients
        loss_dict_mean = {}
        for k, v in loss_dict.items():
            if v is not None:
                if weight_lens is not None:# reweight losses so shorter files adjust the model less than longer files
                    v = v * (weight_lens/weight_lens.sum()) * weight_lens.shape[0]
                loss_dict_mean[k] = dist_mean(v.mean(), world_size).item() if world_size > 1 else v.mean().item()# average tensor along batch dim and all GPUs
        return loss_dict_mean
    
    def get_path_losses(self, loss_dict: dict, loss_dict_g: dict, loss_dict_d: Optional[dict], loss_weights: dict, audiopaths: list, world_size: int):
        loss_dict = deepto(loss_dict, 'cpu')
        if world_size > 1:
            all_audiopaths = [None for _ in range(world_size)]
            dist.all_gather_object(all_audiopaths, audiopaths)
            
            all_loss_dicts = [None for _ in range(world_size)]
            dist.all_gather_object(all_loss_dicts, loss_dict)
        else:
            all_audiopaths = [audiopaths]
            all_loss_dicts = [loss_dict]
        del loss_dict
        all_loss_dicts = [deepto(loss_dict, 'cpu') for loss_dict in all_loss_dicts]
        
        loss_dict_path = {} # {audiopath: {loss_term: loss_value, ...}, ...}
        for audiopaths, l_dict in zip_equal(all_audiopaths, all_loss_dicts):
            for i, audiopath in enumerate(audiopaths):
                loss_dict_path[audiopath] = {}
                for k, v in l_dict.items():
                    loss_dict_path[audiopath][k] = v[i].item()
        
        for audiopath, loss_d in loss_dict_path.items():
            loss_dict_path[audiopath]['loss_total_reduced'] = self.colate_losses(loss_d, loss_weights, silent=True)
            if loss_dict_d is not None:
                loss_dict_path[audiopath]['loss_g_reduced'] = self.colate_losses({k: v for k, v in loss_d.items() if k in loss_dict_g}, loss_weights, silent=True)
                loss_dict_path[audiopath]['loss_d_reduced'] = self.colate_losses({k: v for k, v in loss_d.items() if k in loss_dict_d}, loss_weights, silent=True)
            
            loss_weights_MAE = {k: 0.0 for k, v in loss_weights.items()}
            for k, v in loss_weights.items():
                loss_weights_MAE[k.replace('MSE', 'MAE').replace('NNL', 'MAE')] += v
            loss_dict_path[audiopath]['loss_total_reduced_MAE'] = self.colate_losses(loss_d, loss_weights_MAE, silent=True)
            if loss_dict_d is not None:
                loss_dict_path[audiopath]['loss_g_reduced_MAE'] = self.colate_losses({k: v for k, v in loss_d.items() if k in loss_dict_g}, loss_weights_MAE, silent=True)
                loss_dict_path[audiopath]['loss_d_reduced_MAE'] = self.colate_losses({k: v for k, v in loss_d.items() if k in loss_dict_d}, loss_weights_MAE, silent=True)
        return loss_dict_path
    
    def spkrnames_to_ids(self, spkrnames: Union[list, str], ref_tensor: Optional[torch.Tensor]=None, default_id: Optional[int]=None):
        if isinstance(spkrnames, type('')):
            spkrnames = [spkrnames]
        
        if not hasattr(self, 'spkrlookup'):
            self.spkrlookup = {name: id for name, id, *_ in self.spkrlist}
        
        spkr_ids = []
        for spkrname in spkrnames:
            if spkrname not in self.spkrlookup:
                assert default_id is not None, f'{spkrname} speaker is not in this model'
                spkr_id = default_id
            else:
                spkr_id = self.spkrlookup[spkrname]# lookup ids
            spkr_ids.append(spkr_id)
        
        # collate into tensor
        if ref_tensor is not None:
            spkr_ids = torch.tensor(spkr_ids, device=ref_tensor.device).long()[:, None, None]# [B, 1, 1]
        else:
            spkr_ids = torch.tensor(spkr_ids).long()[:, None, None]# [B, 1, 1]
        return spkr_ids

class ConvNorm(nnModule):
    def __init__(self,
                 in_channels, out_channels, kernel_size=1, stride=1, groups=None, padding=None, dilation=1, bias=True, dropout=0.,
                 w_init_gain=None, act_func=None, act_func_params=None, separable=False,
                 causal=False, pad_right=False, partial_padding=False, ignore_separable_warning=False,
                 LSUV_init=False, n_LSUV_passes=1, LSUV_ignore_act_func=False, LSUV_init_bias=False, w_gain=1.0,
                 weight_norm=False, spectral_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False,
                 pad_value=0.0, allow_linear=True, alpha_dropout=None, always_dropout=False):
        super(ConvNorm, self).__init__()
        if act_func_params is None:
            act_func_params = {}
        if separable and (not ignore_separable_warning):
            assert out_channels//in_channels==out_channels/in_channels, "in_channels must be equal to or a factor of out_channels to use separable Conv1d."
        if not ignore_norm_warning:
            assert bool(instance_norm)+bool(layer_norm)+bool(batch_norm) <= 1, 'only one of instance_norm, layer_norm or batch_norm is recommended to be used at a time. Use ignore_norm_warning=True if you know what you\'re doing'
        if act_func is not None and bias is False:
            print("Warning! Using act_func without any layer bias")
        assert not (spectral_norm and weight_norm), 'can\'t use weight_norm and spectral_norm at the same time'
        self.instance_norm = nn.InstanceNorm1d(out_channels,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_channels, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    = nn.   BatchNorm1d(out_channels,             affine=affine_norm) if    batch_norm else None
        
        self.channel_last_dim = True
        self.partial_padding = partial_padding
        self.pad_value = pad_value
        self.weight_norm = weight_norm
        self.act_func    = act_func
        if type(self.act_func) is str:
            w_init_gain = w_init_gain or get_afunc_gain(self.act_func)[0]
            self.act_func = get_afunc(self.act_func)
        self.act_func_params = act_func_params
        
        if dilation is None:
            dilation = 1
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.separable = (separable and in_channels==out_channels)
        
        self.dropout = dropout
        self.dropout_func = F.alpha_dropout if alpha_dropout or (alpha_dropout is None and act_func == 'selu') else F.dropout
        self.always_dropout = always_dropout
        
        conv_groups = groups or (in_channels if self.separable else 1)
        self.is_linear = kernel_size==1 and conv_groups==1 and allow_linear
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.is_linear:
            self.conv = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels,
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
            self.conv_d = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=bias)
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
    
    def maybe_pad(self, signal, pad_right=False):
        if self.causal_pad:
            if pad_right:
                signal = F.pad(signal, (0, self.causal_pad))
            else:
                signal = F.pad(signal, (self.causal_pad, 0))
        return signal
    
    def pre(self, signal):# [B, C, T] or ([B, T, C] or [B, C])
        self.squeeze_t_dim = False
        if self.channel_last_dim:
            if self.is_linear:
                assert len(signal.shape) in [2, 3], f"input has {len(signal.shape)} dims, should have 2 or 3 for Conv1d/Linear"
                if len(signal.shape) == 2:
                    self.squeeze_t_dim = True
                    signal = signal.unsqueeze(1)# -> [B, T, C]
            else:
                assert len(signal.shape) == 3, f"input has {len(signal.shape)} dims, should have 3 for Conv1d"
            signal = signal.transpose(1, 2)# -> [B, C, T]
        assert signal.shape[1] == self.in_channels, f"input has {signal.shape[1]} channels but expected {self.in_channels}"
        signal = self.maybe_pad(signal, self.pad_right)
        return signal
    
    def conv1(self, signal):
        return self.conv(signal.transpose(1, 2)).transpose(1, 2) if self.is_linear else self.conv(signal)
    
    def main(self, signal, ignore_norm=False, ignore_act_func=False, mask=None):# [B, C, T]
        #assert torch.isfinite(signal).all(), 'got non-finite element in input!'
        conv_signal = self.conv1(signal)
        if self.separable:
            conv_signal = self.conv_d(conv_signal)
        #assert torch.isfinite(signal).all(), 'got non-finite element in output!'
        if self.partial_padding and self.padding:
            # multiply values near the edge by (total edge n_elements/non-padded edge n_elements)
            pad = self.padding
            if mask is None:
                mask = signal.abs().sum(1, True)!=self.pad_value# read zeros in input as masked timesteps
                # [B, 1, T]
            signal_divisor = F.conv1d(mask.to(signal), signal.new_ones((1, 1, self.kernel_size),)/self.kernel_size, padding=pad, stride=self.stride, dilation=self.dilation).clamp_(min=0.0, max=1.0).masked_fill_(~mask[:, :, ::self.stride], 1.0)
            
            if self.conv.bias is not None:
                bias = self.conv.bias.view(1, self.out_channels, 1)# [1, oC, 1]
                conv_signal = conv_signal.sub_(bias).div(signal_divisor).add_(bias).masked_fill_(~mask[:, :, ::self.stride], self.pad_value)
            else:
                conv_signal = conv_signal.div(signal_divisor).masked_fill_(~mask[:, :, ::self.stride], self.pad_value)
        
        if not ignore_norm:
            conv_signal = self.instance_norm(conv_signal) if self.instance_norm is not None else conv_signal
            conv_signal = self.   batch_norm(conv_signal) if self.   batch_norm is not None else conv_signal
            conv_signal = self.   layer_norm(conv_signal
                        .transpose(1, 2)).transpose(1, 2) if self.   layer_norm is not None else conv_signal
        if self.act_func is not None and not ignore_act_func:
            conv_signal = self.act_func(conv_signal, **self.act_func_params)
        if (self.training or self.always_dropout) and self.dropout > 0.:
            conv_signal = self.dropout_func(conv_signal, p=self.dropout, inplace=True, training=True)
        if self.channel_last_dim:
            conv_signal = conv_signal.transpose(1, 2)# -> original shape
        if self.squeeze_t_dim:
            conv_signal = conv_signal.squeeze(1)
        return conv_signal# [B, C, T] or ([B, T, C] or [B, C])
    
    def forward(self, signal):# [B, C, T] or [B, T, C]
        signal = self.pre(signal)# -> [B, C, T+maybe_causal_padding]
        
        if hasattr(self, 'LSUV_init_done') and not self.LSUV_init_done and self.training:
            orig_dropout = self.dropout
            self.dropout = 0.0
            for i in range(self.n_LSUV_passes):
                with torch.no_grad():
                    y = self.main(signal, ignore_norm=True, ignore_act_func=self.LSUV_ignore_act_func)
                    self.conv.weight.data /= y.std()
                    if hasattr(self.conv, 'bias') and self.LSUV_bias:
                        self.conv.bias.data -= y.mean()
            del y
            self.dropout = orig_dropout
            self.LSUV_init_done += True
        
        conv_signal = self.main(signal)
        return conv_signal# [B, C, T] or [B, T, C]

class CondConv(ConvNorm):# Conditional Conv Norm
    def __init__(self, *args, cond_dim=0, mask_zeros=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_zeros = mask_zeros
        if cond_dim:
            out_channels = args[1]
            self.affine = ConvNorm(cond_dim, 2*out_channels)
            self.affine.conv.weight.data *= 0.001
            self.affine.conv.bias.data *= 0.00
    
    def forward(self, signal, cond=None, mask=None):# [B, T, C]
        if self.mask_zeros and mask is None:
            mask = signal.detach().abs().sum(2, keepdim=True) != 0.0
        conv_signal = super().forward(signal)# [B, C, T] or [B, T, C]
        if hasattr(self, 'affine'):
            assert cond is not None
            assert cond.shape[1] == signal.shape[1] or cond.shape[1] == 1, f'got length of {signal.shape[1]}, expected {cond.shape[1]}'
            cond = self.affine(cond)
            if len(conv_signal.shape) > len(cond.shape):
                cond = cond.unsqueeze(1)
            weight, bias = cond.chunk(2, dim=-1)
            weight.data.add_(1.0) # derivative of a constant function is zero, can just modify self.affine output inplace.
            conv_signal = torch.addcmul(bias, conv_signal, weight)# (conv_signal*weight)+bias
        if self.mask_zeros:
            assert conv_signal.shape[1] == mask.shape[1], f'got length of {mask.shape[1]}, expected {conv_signal.shape[1]}'
            conv_signal = conv_signal.masked_fill_(~mask, 0.0)
        return conv_signal


def maybe_invert_mask(mask):# [B, T] force_left_aligned_mask
    if mask.dim() == 3:
        mask = mask.squeeze(2)
    first_mask = mask[:, 0]
    if (~first_mask).all():# if mask's first timestep is not entirely True
        return ~mask
    else:
        return mask

class ResBlock(nnModule):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_blocks: int, n_layers: int, kernel_size: Union[List[int], int] = 1, rezero=False, force_left_aligned_mask=False, skip_all_res=False, **conv_kwargs):
        super(ResBlock, self).__init__()
        self.in_dim     = in_dim
        self.out_dim    = out_dim
        self.hidden_dim = hidden_dim
        self.force_left_aligned_mask = force_left_aligned_mask
        if 'mask_zeros' not in conv_kwargs.keys():
            conv_kwargs['mask_zeros'] = False
        if 'act_func' not in conv_kwargs.keys():
            print('WARNING: act_func is missing from conv_kwargs')
        self.skip_all_res = skip_all_res
        if n_blocks==0 and n_layers==0:
            hidden_dim = in_dim
        if in_dim!=hidden_dim:
            self.pre = CondConv(in_dim, hidden_dim, **{**conv_kwargs, 'separable': False, 'dropout': 0.0})
        self.blocks = nn.ModuleList()
        if type(kernel_size) in [list, tuple]:
            assert len(kernel_size) == n_layers, f'Got {type(kernel_size)} for kernel_size type but length was {len(kernel_size)}, expected length {n_layers} or type of int'
        for block_idx in range(n_blocks):
            convs = nn.ModuleList()
            for layer_idx in range(n_layers):
                ksize = kernel_size[block_idx] if type(kernel_size) in [list, tuple] else kernel_size
                convs.append( CondConv(hidden_dim, hidden_dim, ksize, **conv_kwargs) )
            self.blocks.append(convs)
        if rezero and n_blocks:
            self.rezero = nn.Parameter(torch.ones(n_blocks, dtype=torch.float))
            self.rezero.data *= 1e-3
        if hidden_dim!=out_dim:
            self.post = CondConv(hidden_dim, out_dim, **{**conv_kwargs, 'act_func': None, 'separable': False, 'dropout': 0.0})
    
    def forward(self, x: Tensor, cond: Optional[Tensor] = None, mask: Optional[Tensor] = None, lens: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Optional[Tensor], Tensor]]:# [B, T, C], [B, T, C], [B, T, 1]
        if mask is None and lens is not None:
            mask = get_mask1d(lens, max_len=x.shape[1])
        if self.force_left_aligned_mask and mask is not None:
            mask = maybe_invert_mask(mask)
        if hasattr(self, 'pre'):
            x = self.pre(x, cond)
        if hasattr(self, 'rezero'):
            rezeros = self.rezero.unbind(0)
        for i, block in enumerate(self.blocks):
            x_res = x
            for layer in block:
                x = layer(x, cond)
            
            if not self.skip_all_res:
                if hasattr(self, 'rezero'):
                    x = x*rezeros[i]
                    x = x + x_res
                else:
                    x = (x + x_res)
                    x = x/math.sqrt(2.0)
            
            if mask is not None:
                if len(x.shape) > len(mask.shape):
                    mask = mask.unsqueeze(2)
                x = x.masked_fill_(~mask, 0.0)
        if hasattr(self, 'post'):
            x = self.post(x, cond)
        
        return x# [B, T, C]


class ConvTranspose(nnModule):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=None,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0., act_func=None, act_func_params=None,
                 causal=False, pad_right=False, separable=False, ignore_separable_warning=False, LSUV_init=False, n_LSUV_passes=1, partial_padding=False,
                 weight_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False, pad_value=0.0, allow_linear=True):
        super().__init__()
        if act_func_params is None:
            act_func_params = {}
        if separable and (not ignore_separable_warning):
            assert out_channels//in_channels==out_channels/in_channels, "in_channels must be equal to or a factor of out_channels to use separable Conv1d."
        if not ignore_norm_warning:
            assert bool(instance_norm)+bool(layer_norm)+bool(batch_norm) <= 1, 'only one of instance_norm, layer_norm or batch_norm is recommended to be used at a time. Use ignore_norm_warning=True if you know what you\'re doing'
        self.instance_norm = nn.InstanceNorm1d(out_channels,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_channels, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    = nn.   BatchNorm1d(out_channels,             affine=affine_norm) if    batch_norm else None
        
        self.channel_last_dim = True
        self.partial_padding = partial_padding
        self.pad_value = pad_value
        self.weight_norm = weight_norm
        self.act_func        = act_func
        self.act_func_params = act_func_params
        
        if dilation is None:
            dilation = 1
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.separable = (separable and in_channels==out_channels)
        self.dropout = dropout
        
        conv_groups = groups or (in_channels if self.separable else 1)
        self.is_linear = kernel_size==1 and conv_groups==1 and allow_linear
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose1d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=0 if causal else padding,
                                  dilation=dilation, bias=bias, groups=conv_groups)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain('linear' if self.separable else w_init_gain))
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if self.separable:
            self.conv_d = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=bias)
            torch.nn.init.xavier_uniform_(
                self.conv_d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
            if weight_norm:
                self.conv_d = nn.utils.weight_norm(self.conv_d, name='weight')
        
        self.causal_pad = (kernel_size-stride)*dilation if causal else 0
        self.pad_right  = pad_right
        
        if LSUV_init:
            self.n_LSUV_passes = n_LSUV_passes
            self.register_buffer('LSUV_init_done', torch.tensor(False))
        self.squeeze_t_dim = False
    
    def maybe_pad(self, signal, pad_right=False):
        if self.causal_pad:
            if pad_right:
                signal = F.pad(signal, (0, self.causal_pad))
            else:
                signal = F.pad(signal, (self.causal_pad, 0))
        return signal
    
    def pre(self, signal):# [B, C, T] or ([B, T, C] or [B, C])
        self.squeeze_t_dim = False
        if self.channel_last_dim:
            if self.is_linear:
                assert len(signal.shape) in [2, 3], f"input has {len(signal.shape)} dims, should have 2 or 3 for Conv1d/Linear"
                if len(signal.shape) == 2:
                    self.squeeze_t_dim = True
                    signal = signal.unsqueeze(1)# -> [B, T, C]
            else:
                assert len(signal.shape) == 3, f"input has {len(signal.shape)} dims, should have 3 for Conv1d"
            signal = signal.transpose(1, 2)# -> [B, C, T]
        assert signal.shape[1] == self.in_channels, f"input has {signal.shape[1]} channels but expected {self.in_channels}"
        signal = self.maybe_pad(signal, self.pad_right)
        return signal
    
    def main(self, signal, ignore_norm=False, mask=None):# [B, C, T]
        conv_signal = self.conv(signal)
        if self.separable:
            conv_signal = self.conv_d(conv_signal)
        if self.partial_padding and self.padding:
            # multiply values near the edge by (total edge n_elements/non-padded edge n_elements)
            pad = self.padding
            if mask is None:
                mask = signal.abs().sum(1, True)!=self.pad_value# read zeros in input as masked timesteps
            signal_divisor = F.conv_transpose1d(mask.to(signal), signal.new_ones((1, 1, self.kernel_size),)/self.kernel_size, padding=pad, dilation=self.dilation).clamp_(min=0.0, max=1.0).masked_fill_(~mask, 1.0)
            
            if self.conv.bias is not None:
                bias = self.conv.bias.view(1, self.out_channels, 1)# [1, oC, 1]
                conv_signal = conv_signal.sub_(bias).div(signal_divisor).add_(bias).masked_fill_(~mask, self.pad_value)
            else:
                conv_signal = conv_signal.div(signal_divisor).masked_fill_(~mask, self.pad_value)
        
        if not ignore_norm:
            conv_signal = self.instance_norm(conv_signal) if self.instance_norm is not None else conv_signal
            conv_signal = self.   batch_norm(conv_signal) if self.   batch_norm is not None else conv_signal
            conv_signal = self.   layer_norm(conv_signal
                        .transpose(1, 2)).transpose(1, 2) if self.   layer_norm is not None else conv_signal
        if self.act_func is not None:
            conv_signal = self.act_func(conv_signal, **self.act_func_params)
        if self.training and self.dropout > 0.:
            conv_signal = F.dropout(conv_signal, p=self.dropout, training=self.training)
        if self.channel_last_dim:
            conv_signal = conv_signal.transpose(1, 2)# -> original shape
        if self.squeeze_t_dim:
            conv_signal = conv_signal.squeeze(1)
        return conv_signal# [B, C, T] or ([B, T, C] or [B, C])
    
    def forward(self, signal):# [B, C, T] or [B, T, C]
        signal = self.pre(signal)# -> [B, C, T+maybe_causal_padding]
        if hasattr(self, 'LSUV_init_done') and not self.LSUV_init_done and self.training:
            training = self.training
            self.eval()
            for i in range(self.n_LSUV_passes):
                with torch.no_grad():
                    if self.separable:
                        z = self.conv1(signal)
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
                    z = self.main(signal, ignore_norm=True)
                    if self.separable:
                        self.conv_d.weight.data /= z.std()
                        if hasattr(self.conv_d, 'bias'): self.conv_d.bias.data -= z.mean()
                    else:
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
            del z
            self.train(training)
            self.LSUV_init_done += True
        
        conv_signal = self.main(signal)
        return conv_signal# [B, C, T] or [B, T, C]


def reparameterize(mu: Tensor, logvar: Tensor, training: Union[bool, float]):# use for VAE sampling
    if training:
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        if type(training) is float and 0.0 < training < 1.0:# if training is not True or False:
            eps *= training                                 #   multiply noise by Training factor
        return eps.mul(std).add_(mu)
    else:
        return mu



if __name__ == '__main__':
    layer = ConvNorm(1, 1, kernel_size=3, causal=True)
    
    i = torch.rand(1, 1, 1) # [B, T, C]
    o = layer.forward(i)
    print(o)
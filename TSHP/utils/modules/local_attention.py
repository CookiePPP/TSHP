###########################################################################################
# Code written by lucidrains and taken from https://github.com/lucidrains/local-attention #
###########################################################################################
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from operator import mul
from functools import reduce


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, rand_start_pos=False, period=10000, max_start_pos=16384):
        super().__init__()
        self.dim = dim
        inv_freq = 1. / (period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.rand_start_pos = rand_start_pos
        self.max_start_pos = max_start_pos
    
    def forward(self, x, rand_start_pos=None):# [B, T, D]
        if rand_start_pos is None:
            rand_start_pos = self.rand_start_pos
        if rand_start_pos:
            start_pos = random.randint(0, self.max_start_pos) if self.training else self.max_start_pos//2
        else:
            start_pos = 0
        
        n = x.shape[-2] # [T]
        t = torch.arange(start_pos, start_pos+n, step=1, device=x.device).type_as(self.inv_freq) # [T]
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq) # [T, D//2]
        emb = torch.cat((freqs, freqs), dim=-1)[..., :self.dim] # [T, D]
        return emb[None, :, :] # [1, T, D]

def rotate_half(x):
    #x = x.reshape((x.shape[0], -1, 2, x.shape[-1] // 2)) # [B, T, D] -> [B, T, 2, D]
    x1, x2 = x.split([math.ceil(x.shape[2] / 2), math.floor(x.shape[2] / 2)], 2)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(t, freqs) -> torch.Tensor:
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def apply_rotary_pos_emb_qk(q, k, freqs) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k = map(lambda t: (t * freqs.cos()) + (rotate_half(t) * freqs.sin()), (q, k))
    return q, k

# constant

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work


# helper functions

def exists(val):
    return val is not None


def default(value, d):
    return d if not exists(value) else value


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)


# main class
class LocalAttention(nn.Module):
    def __init__(
            self,
            window_size,
            causal=False,
            look_backward=1,
            look_forward=None,
            dropout=0.,
            shared_qk=False,
            dim=None,
            autopad=True,
            exact_windowsize=False
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        
        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_windowsize = exact_windowsize
        self.autopad = autopad
        
        self.dropout = nn.Dropout(dropout)
        
        self.shared_qk = shared_qk
        
        self.rel_pos = None
        if exists(dim):
            self.rel_pos = SinusoidalEmbeddings(dim)
    
    def forward(self, q, k, v, input_mask=None):# [B, T, D]
        shape = q.shape
        
        def merge_into_batch(t):
            return t.reshape(-1, *t.shape[-2:])
        
        q, k, v = map(merge_into_batch, (q, k, v))
        
        if exists(self.rel_pos):
            pos_emb = self.rel_pos(q)
            q, k = apply_rotary_pos_emb_qk(q, k, pos_emb)
        
        orig_t = q.shape[1]
        if self.autopad:
            q, k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v))
        
        (b, T, e), device, dtype = q.shape, q.device, q.dtype
        assert (T % self.window_size) == 0, f'sequence length {T} must be divisible by window size {self.window_size} for local attention'
        
        windows = T // self.window_size
        
        if self.shared_qk:
            k = F.normalize(k, 2, dim=-1).type_as(q)
        
        ticker = torch.arange(T, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, self.window_size)
        
        def bucket_fn(t):
            return t.reshape(b, windows, self.window_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))
        
        look_around_kwargs = {'backward': self.look_backward, 'forward': self.look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        
        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        
        dots = (torch.einsum('bhie,bhje->bhij', bq.float(), bk.float())*(e ** -0.5)).to(bq)
        
        mask_value = max_neg_value(dots)
        
        if self.shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask
    
        if self.causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
        
            if self.exact_windowsize:
                max_causal_window_size = (self.window_size * self.look_backward)
                mask = mask | (bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size))
        
            dots.masked_fill_(mask, mask_value)
            del mask
        
        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask
        
        if input_mask is not None:
            input_mask = input_mask.view(input_mask.shape[0], input_mask.shape[1], 1).squeeze(2)
            h = b // input_mask.shape[0]
            if self.autopad:
                input_mask = pad_to_multiple(input_mask, self.window_size, dim=-1, value=False)
            input_mask = input_mask.reshape(-1, windows, self.window_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = (mq[:, :, :, None] * mk[:, :, None, :])
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask
        
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(-1, T, e)
        
        if self.autopad:
            out = out[:, :orig_t, :]
        
        return out.reshape(*shape)

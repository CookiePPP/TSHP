from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple, Optional, Union


# get_mask_from_lengths()
# get_mask_3d()
# maybe_cat()
# grad_cp()
# Fpad()
# dropout_frame()
# dropout_batch()

def get_mask(lengths: Tensor, max_len: int = 0) -> torch.BoolTensor:# [B, ...]
    return get_mask1d(lengths, max_len, squ=True)# -> [B, T]

def get_mask1d(lengths: Tensor, max_len: int = 0, squ=False):# [B, ...]
    assert isinstance(lengths, Tensor), f'input is type {type(lengths)}, expected Tensor'
    lengths = lengths.view(-1)# [B]
    if max_len == 0:
        max_len = int(torch.max(lengths).item())
    ids : torch.LongTensor = torch.arange(0, max_len, device=lengths.device, dtype=torch.long).unsqueeze(0)
    mask: torch.BoolTensor = ids < lengths.unsqueeze(1)# [B, T]
    if squ:
        return mask# [B, T]
    return mask.unsqueeze(2)# [B, T, 1]


def get_mask2d(lengths1: Tensor, lengths2: Tensor, max_w: Optional[Union[int, Tensor]] = None, max_h: Optional[Tensor] = None) -> torch.BoolTensor:
    device = lengths1.device
    if max_w is None:
        max_w = torch.max(lengths1).item()
    if max_h is None:
        max_h = torch.max(lengths2).item()
    seq_w = torch.arange(0, max_w, device=device)# [max_w]
    seq_h = torch.arange(0, max_h, device=device)# [max_h]
    mask_w = (seq_w.unsqueeze(0) < lengths1.unsqueeze(1)).to(torch.bool)# [1, max_w] < [B, 1] -> [B, max_w]
    mask_h = (seq_h.unsqueeze(0) < lengths2.unsqueeze(1)).to(torch.bool)# [1, max_h] < [B, 1] -> [B, max_h]
    mask: torch.BoolTensor = (mask_w.unsqueeze(2) & mask_h.unsqueeze(1))# [B, max_w, 1] & [B, 1, max_h] -> [B, max_w, max_h]
    return mask# [B, max_w, max_h]

def maybe_cat(*tensors, dim: int = -1, pad_to_common_lengths=False, length_dim=1, pad_val=0.0) -> Tensor:
    """
    
    @param tensors: List of Tensors to be concatenated
    @param dim: the dimension to concatenate each dim along
    @return: concatenated Tensor
    """
    if len(tensors) == 1 and type(tensors[0]) in [list, tuple]:
        tensors = tensors[0]
    
    # ignore None elements
    tensors = [t for t in tensors if t is not None]
    
    # (optional) pad to common length
    if pad_to_common_lengths:
        tensors = pad_to_common(*tensors, length_dim=length_dim, pad_val=pad_val)
    
    out = []
    max_dims = list(max(x.shape[i] for x in tensors) for i in range(tensors[0].dim()))
    max_dims[dim] = -1
    for i, tensor in enumerate(tensors):
        if not type(tensor) is Tensor:
            raise TypeError(f'got type {type(tensor)} on index {i}, expected Tensor')
        tensor = tensor.expand(*max_dims)# expand any [1, 1, 1] to the max dim that was found in all tensors. Lets you add [1] dims to [?] dims
        out.append(tensor)
    if len(out) > 1:
        out = torch.cat(out, dim=dim)
    else:
        out = out[0]
    return out


def pad_to_common(*tensors: Tensor, length_dim: int = 1, pad_val: float = 0.0):
    if len(tensors) == 1 and type(tensors[0]) in [list, tuple]:
        tensors = tensors[0]
    
    max_T = max(t.shape[length_dim] for t in tensors)
    tensors: List[Tensor] = [Fpad(t, (0, max_T - t.shape[length_dim]), value=pad_val) for t in tensors]
    return tensors


# grad_cp() # checkpoint with proper kwarg support and dict/list/tuple support


# Fpad() # padding with [B, T, D] support
def Fpad(*args, **kwargs):
    args = list(args)
    x_has_2_dim = kwargs.pop('has_2dim', True)
    if not x_has_2_dim and args[0].dim() == 2:
        args[0] = args[0][..., None]
    
    args = [args[0].transpose(-1, -2), *args[1:]]# [..., T, D] -> [..., D, T]
    x: Tensor = F.pad(*args, **kwargs)
    x = x.transpose(-1, -2)# [..., D, T] -> [..., T, D]
    
    if not x_has_2_dim:
        x = x.squeeze(2)
    return x

def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):# [B, T, C]
    return F.avg_pool1d(input.transpose(1, 2), kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad).transpose(1, 2)

def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate, max_len, chunk_size=1):
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = int(torch.max(lengths).item())
    mask = get_mask1d(lengths, max_len)
    drop_mask: torch.BoolTensor = torch.empty([batch_size, ceil(max_len/chunk_size), 1], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask.repeat_interleave(chunk_size, dim=1)[:, :max_len] * mask
    return drop_mask


# mask random frames with local means of the neighboring frames channels
def dropout_frame(mel, mel_lengths, drop_frame_rate: float, soft_mask=False, smooth_range: int = 9, max_len=None, chunk_size=1):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate, max_len=max_len, chunk_size=chunk_size)# [B, mel_T, 1]
    
    with torch.no_grad():
        weight = torch.ones(mel.shape[2], 1, smooth_range).to(mel)/smooth_range
        smooth_range_sub1 = smooth_range-1
        smoothed_mel = F.conv1d(Fpad(mel, (smooth_range_sub1//2, -(-smooth_range_sub1//2)), mode='replicate').transpose(1, 2), weight, groups=mel.shape[2]).transpose(1, 2)
    
    dropped_mels = (mel * ~drop_mask) + (smoothed_mel * drop_mask)
    if soft_mask:
        rand_mask = torch.rand(dropped_mels.shape[0], dropped_mels.shape[1], 1, device=mel.device, dtype=mel.dtype)# [B, T, 1]
        rand_mask_inv = (1.-rand_mask)
        dropped_mels = (dropped_mels*rand_mask) + (mel * rand_mask_inv)
    return dropped_mels

# dropout_batch() # dropout with [B, ...] mask

def get_first_over_thresh(x, threshold):
    """Takes [B, T] and outputs indexes of first element over threshold for each vector in B (output.shape = [B])."""
    device = x.device
    x = x.clone().cpu().float() # using CPU because GPU implementation of argmax() splits tensor into 32 elem chunks, each chunk is parsed forward then the outputs are collected together... backwards
    x[:, -1] = threshold # set last to threshold just incase the output didn't finish generating.
    x[x>threshold] = threshold
    if int(''.join(torch.__version__.split('+')[0].split('.'))) < 170:# old pytorch < 1.7 did argmax backwards on CPU and even more broken on GPU.
        return ( (x.size(1)-1)-(x.flip(dims=(1,)).argmax(dim=1)) ).to(device).int()
    else:
        return x.argmax(dim=1).to(device).int()


def long_to_bits(x, bits):
    mask = 2**torch.arange(bits-1, device=x.device, dtype=x.dtype)
    sign_bit = x.sign() > -0.5
    return torch.cat([sign_bit.unsqueeze(-1), x.abs().unsqueeze(-1).bitwise_and(mask).ne(0).byte()], dim=-1)

def bits_to_long(x):
    mask = 2**torch.arange(x.shape[-1]-1, device=x.device, dtype=x.dtype)
    sign = x[..., 0].long()
    sign[sign==0] = -1
    return (x[..., 1:]*mask).sum(-1, dtype=torch.long) * sign


def diagonal_pad(x):# [B, T, C]
    """
    Pads each channel with it's index number of zeros on the front.
      1 2 2
      1 2 2
      1 2 2
    goes to
    1 2 2 0 0
    0 1 2 2 0
    0 0 1 2 2
    
    
    Pads each channel with it's index number of zeros on the front.
      1 2 2 2
      1 2 2 2
      1 2 2 2
      1 2 2 2
    goes to
    1 2 2 2 0 0 0
    0 1 2 2 2 0 0
    0 0 1 2 2 2 0
    0 0 0 1 2 2 2
    
    """
    out_x = []
    C = x.shape[2]
    if C == 1:
        return x
    for i, c in enumerate(x.unbind(2)):# (int, [B, T])
        c = F.pad(c[:, None, :], (i, C -1 -i)).squeeze(1)
        out_x.append(c)
    x = torch.stack(out_x, dim=2)
    return x

def remove_diagonal_pad(x):# [B, T, C, ...]
    """
    1 2 2 0 0
    0 1 2 2 0
    0 0 1 2 2
    goes to
    1 2 2
    1 2 2
    1 2 2
    """
    out_x = []
    C = x.shape[2]
    if C == 1:
        return x
    for i, c in enumerate(x.unbind(2)):# (int, [B, T, ...])
        end_pad = C -1 -i
        if end_pad > 0:# [:-0] will give an empty list so this if statement
            c = c[:, i:-end_pad]
        else:
            c = c[:, i:]
        out_x.append(c)
    x = torch.stack(out_x, dim=2)# [B, T, C, ...]
    return x


def align_duration(seq,                    # [B, txt_T,     C]
                   seq_dur,                # [B, txt_T,     1]
                   smoothing_order=0,      # int
                   mel_lens=None,          # [B,     1,     1]
                   seq_lengths=None,       # [B,     1,     1]
                   attention_override=None # [B, txt_T, mel_T]
                   ):
    
    if attention_override is None:
        B, txt_T, C = seq.shape# [B, Text Length, Encoder Dimension]
        
        if mel_lens is None:# get Length of output Tensors
            mel_lens = seq_dur.sum(1, keepdim=True).floor()# [B, 1, 1]
        mel_T = mel_lens.max().item()
        
        seq_dur = seq_dur.squeeze(2)# -> [B, txt_T]
        seq_lengths = seq_lengths.view(-1)
        
        start_pos     = torch.zeros (B,               device=seq.device, dtype=seq.dtype)# [B]
        attention_pos = torch.arange(mel_T,           device=seq.device, dtype=seq.dtype).expand(B, mel_T)# [B, mel_T]
        attention     = torch.zeros (B, mel_T, txt_T, device=seq.device, dtype=seq.dtype)# [B, mel_T, txt_T]
        for enc_inx in range(seq_dur.shape[1]):
            dur = seq_dur[:, enc_inx]# [B]
            end_pos = start_pos + dur# [B]
            if seq_lengths is not None:# if last char in seq, extend this duration till end of non-padded area.
                mask = (seq_lengths == (enc_inx+1))# [B]
                if mask.any():
                    end_pos.masked_fill_(mask, mel_T)
            
            att = (attention_pos>=start_pos.unsqueeze(-1).repeat(1, mel_T)) & (attention_pos<end_pos.unsqueeze(-1).repeat(1, mel_T))
            attention[:, :, enc_inx][att] = 1.# set predicted duration values to positive
            
            start_pos = start_pos + dur # [B]
        
        for i in range(smoothing_order):
            pad = (-1, 1) if i%2==0 else (1, -1)
            attention += F.pad(attention.transpose(1, 2), pad, mode='replicate').transpose(1, 2)# [B, mel_T, txt_T]
            attention /= 2
    else:
        attention = attention_override
    return attention@seq, attention.transpose(1, 2)# [B, mel_T, txt_T] @ [B, txt_T, C] -> [B, mel_T, C], [B, txt_T, mel_T]


class freeze_grads:
    def __init__(self, submodule):
        self.submodule = submodule
    
    def __enter__(self):
        self.require_grads = []
        for param in self.submodule.parameters():
            self.require_grads.append(param.requires_grad)
            param.requires_grad = False
    
    def __exit__(self, type_, value, traceback):
        for i, param in enumerate(self.submodule.parameters()):
            param.requires_grad = self.require_grads[i]

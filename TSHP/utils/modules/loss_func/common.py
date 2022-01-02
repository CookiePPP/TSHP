from typing import Tuple, Union, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import functional as F

from TSHP.utils.modules.RevGrad import ScaleGrad
from TSHP.utils.modules.utils import get_mask1d

def get_gate_BCE(pr_gate_logits: Tensor, mel_lens: Tensor, pos_weight: float) -> Tensor:
    """
    @param pr_gate_logits: predicted gate values of shape [B, T, 1] with range [-inf, inf] 
    @param mel_lens: lengths of each item, shape [B, ...]
    @param pos_weight: float, higher values increase weighting of positive class
    @return: gate_BCE: FloatTensor[B]
    """
    assert mel_lens.min() > 0
    mask_minus1 = get_mask1d(mel_lens -1, max_len=pr_gate_logits.shape[1])
    gt_gate = torch.logical_xor(mask_minus1, get_mask1d(mel_lens, max_len=pr_gate_logits.shape[1]))# [B, mel_T, 1]
    gate_BCE: torch.Tensor = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='none')(pr_gate_logits, gt_gate.float())
    gate_BCE = gate_BCE.masked_fill_(~get_mask1d(mel_lens, max_len=pr_gate_logits.shape[1]), 0.0)
    gate_BCE = gate_BCE.sum([1, 2], dtype=torch.float)/mel_lens.squeeze()# sum([B, mel_T, 1], dim=[1, 2]) / [B] -> [B]
    return gate_BCE

def get_mean_errors(gt_X: Union[Tensor, float], pr_X: Union[Tensor, float], lens: Tensor = None, mask: Tensor = None, mask_gt_zeros: bool = False, rescale_grad: bool = False) -> Tuple[Tensor, Tensor]:
    """
    @param rescale_grad: rescale gradients so gradient noise is constant regardless of input length
    @param gt_X: Target Tensor of shape [B, T, C]
    @param pr_X: Predicted Tensor of shape [B, T, C]
    @param lens: Lengths of non-padded items, of shape [B, ...]
    @return: MAE, MSE (of shape [B], [B])
    """
    assert lens is None or lens.min() > 0, f'got lens.min() of {lens.min().item()}, expected greater than 0'
    C: int = 1
    if isinstance(gt_X, Tensor) and gt_X.dim() == 3:
        C = max(C, gt_X.shape[2])
    if isinstance(pr_X, Tensor) and pr_X.dim() == 3:
        C = max(C, pr_X.shape[2])
    if type(gt_X) in [int, float]:
        gt_X = torch.tensor(gt_X, device=pr_X.device, dtype=pr_X.dtype)
    assert isinstance(gt_X, Tensor)
    assert isinstance(pr_X, Tensor)
    gt_X, pr_X = torch.broadcast_tensors(gt_X, pr_X)# manually broadcast_tensors to prevent skip pytorch warning
    assert gt_X.shape[0] == pr_X.shape[0] or gt_X.shape[0] == 1 or pr_X.shape[0] == 1, f'got {gt_X.shape[0]} and {pr_X.shape[0]} for shape[0]'
    assert gt_X.shape[1] == pr_X.shape[1] or gt_X.shape[1] == 1 or pr_X.shape[1] == 1, f'got {gt_X.shape[1]} and {pr_X.shape[1]} for shape[1]'
    assert gt_X.shape[2] == pr_X.shape[2] or gt_X.shape[2] == 1 or pr_X.shape[2] == 1, f'got {gt_X.shape[2]} and {pr_X.shape[2]} for shape[2]'
    
    if mask is None:
        if lens is not None:
            assert isinstance(lens, Tensor)
            lens = lens.squeeze()
            mask = get_mask1d(lens, max_len=gt_X.shape[1])
        else:
            mask: Tensor = gt_X[:, :, :1].eq(gt_X[:, :, :1])
            lens = mask.sum(1).view(-1)
    
    if mask_gt_zeros:
        mask = mask & (gt_X.sum(2, keepdim=True)!=0.0) # mask any elements that are zero in gt_X
    MAE: torch.Tensor = F. l1_loss(gt_X, pr_X, reduction='none')
    MAE: torch.Tensor = MAE.masked_fill_(~mask, 0.0).sum([1, 2], dtype=torch.float)/(lens*C)# -> [B]
    MSE: torch.Tensor = F.mse_loss(gt_X, pr_X, reduction='none')
    MSE: torch.Tensor = MSE.masked_fill_(~mask, 0.0).sum([1, 2], dtype=torch.float)/(lens*C)# -> [B]
    assert MAE.shape[0] == gt_X.shape[0]
    assert MSE.shape[0] == gt_X.shape[0]
    if rescale_grad:
        grad_scale = (lens*C)**0.5 # [B]
        MAE = ScaleGrad(grad_scale)(MAE)
        MSE = ScaleGrad(grad_scale)(MSE)
    return MAE, MSE

def get_CTC_loss(log_probs, target_ids, prob_lens, target_lens, blank_idx=0, zero_infinity=True):
    """
    The Connectionist Temporal Classification loss.
    
    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the probability of possible alignments of input to target, producing a loss value which is differentiable with respect to each input node. The alignment of input to target is assumed to be “many-to-one”, which limits the length of the target sequence such that it must be \leq≤ the input length.
    
    trg_T must be less than src_T
    
    target_ids must be shorter than log_probs
    
    @param log_probs  : log probabilties FloatTensor [B, src_T, n_classes]
    @param target_ids : LongTensor [B, trg_T, 1] where values are in range [0, n_classes-1]
    @param prob_lens  : LongTensor [B, 1, 1] where .max() == src_T
    @param target_lens: LongTensor [B, 1, 1] where .max() == trg_T
    @param blank_idx  : int - this idx is predicted to indicate the gap between repeating ids, e.g: if the target is 'BB' then the model should predict 'BBBB{blank}BBBB' to tell it appart from 'BBBBBBBB' which would decode to 'B'
    @return: FloatTensor[B]
    """
    log_probs = log_probs.transpose(0, 1).float() # [B, trg_T, n_classes] -> [trg_T, B, n_classes]
    target_ids = target_ids.view(target_ids.shape[0], target_ids.shape[1], 1).squeeze(2) # ensure targets is correct shape
    prob_lens = prob_lens.view(-1)
    target_lens = target_lens.view(-1)
    
    loss = F.ctc_loss(log_probs, target_ids, prob_lens, target_lens, blank=blank_idx, reduction='none', zero_infinity=zero_infinity).div(target_lens.view(-1))
    return loss # [B]

def get_BCE(gt_X: Union[Tensor, float], pr_X: Union[Tensor, float], lens: Tensor = None, mask: Tensor = None, rescale_grad: bool = False, ignore_out_of_range: bool = False) -> Tuple[Tensor, Tensor]:
    """
    @param gt_X: Target Tensor of shape [B, T, C]
    @param pr_X: Predicted Tensor of shape [B, T, C]
    @param lens: Lengths of non-padded items, of shape [B, ...]
    @return: MAE, MSE (of shape [B], [B])
    """
    # check inputs
    assert lens is None or lens.min() > 0, f'got lens.min() of {lens.min().item()}, expected greater than 0'
    C: int = 1
    if isinstance(gt_X, Tensor) and gt_X.dim() == 3:
        C = max(C, gt_X.shape[2])
    if isinstance(pr_X, Tensor) and pr_X.dim() == 3:
        C = max(C, pr_X.shape[2])
    if type(gt_X) in [int, float]:
        gt_X = torch.tensor(gt_X, device=pr_X.device, dtype=pr_X.dtype)
    assert isinstance(gt_X, Tensor)
    assert isinstance(pr_X, Tensor)
    gt_X, pr_X = torch.broadcast_tensors(gt_X, pr_X)# manually broadcast_tensors to prevent skip pytorch warning
    assert gt_X.shape[0] == pr_X.shape[0] or gt_X.shape[0] == 1 or pr_X.shape[0] == 1, f'got {gt_X.shape[0]} and {pr_X.shape[0]} for shape[0]'
    assert gt_X.shape[1] == pr_X.shape[1] or gt_X.shape[1] == 1 or pr_X.shape[1] == 1, f'got {gt_X.shape[1]} and {pr_X.shape[1]} for shape[1]'
    assert gt_X.shape[2] == pr_X.shape[2] or gt_X.shape[2] == 1 or pr_X.shape[2] == 1, f'got {gt_X.shape[2]} and {pr_X.shape[2]} for shape[2]'
    
    if mask is None:
        if lens is not None:
            assert isinstance(lens, Tensor)
            lens = lens.squeeze()
            mask = get_mask1d(lens, max_len=gt_X.shape[1])
        else:
            mask: Tensor = gt_X[:, :, :1]==gt_X[:, :, :1]
            lens = torch.tensor([mask.shape[1],]*mask.shape[0], device=gt_X.device, dtype=torch.long)
    gt_X = gt_X.masked_fill(~mask, 0.0)
    pr_X = pr_X.masked_fill(~mask, 0.0)
    
    # check for out-of-range/non-finite elements
    if ignore_out_of_range:
        # [-2.0, -1.0, 0.0, 1.0, 2.0]
        gt_X.data.clamp_(min=0.0, max=1.0)
    
    assert gt_X.min() >= 0.0, f'target tensor has min() of {gt_X.min().item()}, expected 0, 1 or inbetween'
    assert gt_X.max() <= 1.0, f'target tensor has max() of {gt_X.max().item()}, expected 0, 1 or inbetween'
    assert pr_X.min() >= 0.0,   f'pred tensor has min() of {gt_X.min().item()}, expected 0, 1 or inbetween'
    assert pr_X.max() <= 1.0,   f'pred tensor has max() of {gt_X.max().item()}, expected 0, 1 or inbetween'
    
    # Calc Mean Binary Cross Entropy of each item in batch
    BCE: torch.Tensor = F.binary_cross_entropy(gt_X, pr_X, reduction='none')
    BCE: torch.Tensor = BCE.masked_fill_(~mask, 0.0).sum([1, 2], dtype=torch.float)/(lens*C)# -> [B]
    assert BCE.shape[0] == gt_X.shape[0]
    
    # Calc Pred Accuracy of each item in batch assuming all gt_X is bool, and 0.5 is the threshold for positive pred output.
    ACC: torch.Tensor = gt_X.round().eq(pr_X.round()).float().masked_fill(~mask, 0.0).sum([1, 2], dtype=torch.float)/(lens*C)# [B, T, 1]
    assert ACC.shape[0] == gt_X.shape[0]
    
    if rescale_grad:
        grad_scale = (lens*C)**0.5 # [B]
        BCE = ScaleGrad(grad_scale)(BCE)
        ACC = ScaleGrad(grad_scale)(ACC)
    return BCE, ACC


def get_GAN_classification_losses(fakeness: Union[Tuple, List, Tensor], trgt: float, lens=None, mask=None,
                                  list_reduction='mean', list_grad_reduction='mean', channel_grad_reduction='mean'):
    if type(fakeness) in [tuple, list]:
        trgt_tensor = torch.tensor(trgt, device=fakeness[0].device, dtype=fakeness[0].dtype)# use real value for fake target
        
        lens_i = lens[0] if type(lens) in [tuple, list] else lens
        mask_i = mask[0] if type(mask) in [tuple, list] else mask
        MAE, MSE = get_mean_errors(fakeness[0], trgt_tensor, lens=lens_i, mask=mask_i)
        BCE, ACC = get_BCE        (fakeness[0], trgt_tensor, lens=lens_i, mask=mask_i, ignore_out_of_range=True)
        
        for i, fakeness_i in list(enumerate(fakeness))[1:]:
            lens_i = lens[i] if type(lens) in [tuple, list] else lens
            mask_i = mask[i] if type(mask) in [tuple, list] else mask
            
            MAE_i, MSE_i = get_mean_errors(fakeness_i, trgt_tensor, lens=lens_i, mask=mask_i)
            BCE_i, ACC_i = get_BCE        (fakeness_i, trgt_tensor, lens=lens_i, mask=mask_i, ignore_out_of_range=True)
            MAE = MAE + MAE_i
            MSE = MSE + MSE_i
            BCE = BCE + BCE_i
            ACC = ACC + ACC_i
        if list_reduction == 'sum':
            assert list_grad_reduction == 'sum'
        elif list_reduction == 'mean':
            MAE = MAE / len(fakeness)
            MSE = MSE / len(fakeness)
            BCE = BCE / len(fakeness)
            ACC = ACC / len(fakeness)
            if list_grad_reduction == 'sum':
                MAE, MSE, BCE, ACC = [ScaleGrad(len(fakeness))(x) for x in [MAE, MSE, BCE, ACC]]
            elif list_grad_reduction is None or list_grad_reduction == 'mean':
                pass
            else:
                raise NotImplementedError(f'list_grad_reduction = {list_grad_reduction} is not valid/supported. Expected \'mean\' or \'sum\'')
        else:
            raise NotImplementedError(f'list_reduction = {list_reduction} is not valid/supported. Expected \'mean\' or \'sum\'')
    elif type(fakeness) in [Tensor, ]:
        trgt_tensor = torch.tensor(trgt, device=fakeness.device, dtype=fakeness.dtype)# use real value for fake target
        MAE, MSE = get_mean_errors(fakeness, trgt_tensor, lens=lens, mask=mask)
        BCE, ACC = get_BCE        (fakeness, trgt_tensor, lens=lens, mask=mask, ignore_out_of_range=True)
    else:
        raise NotImplementedError(f'got input types {type(fakeness)}, expected Tuple or Tensor')
    
    if channel_grad_reduction == 'sum' and fakeness.shape[2] != 1:
        MAE, MSE, BCE, ACC = [ScaleGrad(fakeness.shape[2])(x) for x in [MAE, MSE, BCE, ACC]]
    elif channel_grad_reduction is None or channel_grad_reduction == 'mean':
        pass
    else:
        raise NotImplementedError(f'channel_grad_reduction = {channel_grad_reduction} is not implemented')
    return MAE, MSE, BCE, ACC

def GAN_gloss(fake_fakeness: Union[Tuple, List, Tensor], lens=None, mask=None, channel_grad_reduction='mean', list_grad_reduction='mean', list_reduction='mean'):# [B, T, 1], [B, 1, 1], [B, T, 1]
    """
    fake_fakeness is the predicted 'fakeness' of a fake input sample.
    """
    real_trgt = 0.0
    MAE, MSE, BCE, ACC = get_GAN_classification_losses(fake_fakeness, trgt=real_trgt, lens=lens, mask=mask, list_grad_reduction=list_grad_reduction, list_reduction=list_reduction, channel_grad_reduction=channel_grad_reduction)
    return MAE, MSE, BCE, ACC

def GAN_dloss(real_fakeness: Union[Tuple, List, Tensor],# [B, T, 1]
              fake_fakeness: Union[Tuple, List, Tensor],# [B, T, 1]
              lens     : Optional[Tensor] = None, mask     : Optional[Tensor] = None,# [B, 1, 1]
              fake_lens: Optional[Tensor] = None, fake_mask: Optional[Tensor] = None,# [B, 1, 1]
              real_lens: Optional[Tensor] = None, real_mask: Optional[Tensor] = None,# [B, 1, 1]
              channel_grad_reduction='mean', list_grad_reduction='mean', list_reduction='mean',
              simplified_output=True):
    """
    real_fakeness is the predicted 'fakeness' of a real input sample.
    fake_fakeness is the predicted 'fakeness' of a fake input sample.
    
    simplified_output:bool - average fake+real loss terms, e.g: (f_MAE+r_MAE)/2. instead of f_MAE and r_MAE separately.
    """
    if fake_lens is None:
        fake_lens = lens
    if real_lens is None:
        real_lens = lens
    if fake_mask is None:
        fake_mask = mask
    if real_mask is None:
        real_mask = mask
    
    fake_trgt = 1.0
    f_MAE, f_MSE, f_BCE, f_ACC = get_GAN_classification_losses(
        fake_fakeness, trgt=fake_trgt, lens=fake_lens, mask=fake_mask,
        list_grad_reduction=list_grad_reduction, list_reduction=list_reduction,
        channel_grad_reduction=channel_grad_reduction)
    
    real_trgt = 0.0
    r_MAE, r_MSE, r_BCE, r_ACC = get_GAN_classification_losses(
        real_fakeness, trgt=real_trgt, lens=real_lens, mask=real_mask,
        list_grad_reduction=list_grad_reduction, list_reduction=list_reduction,
        channel_grad_reduction=channel_grad_reduction)
    
    if simplified_output:
        MAE = (f_MAE+r_MAE)/2. # average the fake and real terms
        MSE = (f_MSE+r_MSE)/2. # average the fake and real terms
        BCE = (f_BCE+r_BCE)/2. # average the fake and real terms
        ACC = (f_ACC+r_ACC)/2. # average the fake and real terms
        return MAE, MSE, BCE, ACC
    else:
        return f_MAE, f_MSE, f_BCE, f_ACC, r_MAE, r_MSE, r_BCE, r_ACC


def get_KLD_loss(mu: Tensor, logvar: Tensor,
                 trgt_mu: Optional[Tensor] = None, trgt_logvar: Optional[Tensor] = None,
                 lens   : Optional[Tensor] = None, mask       : Optional[Tensor] = None,
                 rescale_grad: bool = False):
    C = mu.shape[2]
    kld = kld_loss(mu, logvar, trgt_mu, trgt_logvar)# [B, T, 1]
    if mask is None and lens is not None:
        mask = get_mask1d(lens)
    if lens is None and mask is not None:
        lens = mask.sum(1).view(-1).long()
    kld = kld.masked_fill(~mask, 0.0).sum([1, 2], dtype=torch.float)/(lens.view(-1)*C)# -> [B]
    
    if rescale_grad:
        grad_scale = (lens*C)**0.5 # [B]
        kld = ScaleGrad(grad_scale)(kld)
    return kld


def kld_loss(mu1: Tensor, logvar1: Tensor, mu2: Optional[Tensor] = None, logvar2: Optional[Tensor] = None, normal_weight: float = 0.01) -> Tensor:
    if mu2 is None and logvar2 is None:
        var1 = logvar1.float().exp()
        mu_mse = mu1.pow(2)
        logsigma1 = logvar1.mul(0.5)
        kld = -0.5 -logsigma1 + 0.5 * (var1 + mu_mse) # kld_loss against N~(0, 1)
        # [..., D] -> [..., 1]
        return kld.sum(-1, dtype=torch.float, keepdim=True)
    elif mu2 is not None and logvar2 is not None:
        var1 = logvar1.exp()
        logsigma1 = logvar1.mul(0.5)
        var2 = logvar2.exp()
        logsigma2 = logvar2.mul(0.5)
        mu_mse = F.mse_loss(mu1, mu2, reduction='none')
        kld = -0.5 +logsigma2 -logsigma1 + 0.5 * (var1 + mu_mse) / var2
        
        if normal_weight:
            kld = kld + ( -0.5 -logsigma1 + 0.5 * (var1 + mu1.pow(2)) ) # kld_loss against N~(0, 1)
            kld = kld + ( -0.5 -logsigma2 + 0.5 * (var2 + mu2.pow(2)) ) # kld_loss against N~(0, 1)
        return kld.sum(-1, dtype=torch.float, keepdim=True)
    else:
        raise NotImplementedError(f'{mu1 is not None} {logvar1 is not None} {mu2 is not None} {logvar2 is not None}')


def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    sigma1 = logsigma1.exp()
    var1 = sigma1.pow(2)
    sigma2 = logsigma2.exp()
    var2 = sigma2.pow(2)
    mu_mse = F.mse_loss(mu1, mu2, reduction='none')
    return -0.5 +logsigma2 -logsigma1 + 0.5 * (var1 + mu_mse) / var2

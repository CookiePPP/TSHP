from typing import Optional, Tuple

import torch

# https://github.com/gothiswaysir/Transformer_Multi_encoder/blob/952868b01d5e077657a036ced04933ce53dcbf4c/nets/pytorch_backend/e2e_tts_tacotron2.py#L28-L156
from torch import Tensor

from CookieSpeech.utils.misc_utils import zip_equal

from CookieSpeech.utils.modules.utils import get_mask


class GuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.
    This module calculates the guided attention loss described in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_, which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """
    guided_attn_masks: Optional[Tensor] = None
    masks: Optional[Tensor] = None
    def __init__(self,
                 sigma: float = 0.4,
                 reset_always: bool = True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control how close attention to a diagonal.
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.reset_always = reset_always
    
    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None
    
    def forward(self, att_ws: Tensor, ilens: torch.LongTensor, olens: torch.LongTensor) -> Tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights [B, T_out, T_in]
            ilens (LongTensor): Batch of input lengths [B]
            olens (LongTensor): Batch of output lengths [B]
        Returns:
            Tensor: Guided attention loss values [B]
        """
        ilens = ilens.view(-1)
        olens = olens.view(-1)
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        B, out_T, in_T = self.guided_attn_masks.shape
        l1_err = self.guided_attn_masks * att_ws[:, :out_T, :in_T]# [B, out_T, in_T]
        l1_err = l1_err.masked_fill(~self.masks.bool(), 0.0)# mask loss where padded
        l2_err = l1_err.pow(2)
        l1_mae: Tensor = l1_err.sum([1, 2])/olens # [B, out_T, in_T] -> [B]
        l2_mse: Tensor = l2_err.sum([1, 2])/olens # [B, out_T, in_T] -> [B]
        if self.reset_always:
            self._reset_masks()
        return l1_mae, l2_mse# [B], [B]
    
    def _make_guided_attention_masks(self, ilens: Tensor, olens: Tensor) -> Tensor:
        n_batches = ilens.shape[0]
        max_ilen = int(ilens.max().item())
        max_olen = int(olens.max().item())
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=ilens.device)
        for idx, (ilen, olen) in enumerate(zip_equal(ilens.view(-1), olens.view(-1))):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(ilen, olen, self.sigma)
        return guided_attn_masks
    
    @staticmethod
    def _make_guided_attention_mask(ilen: Tensor, olen: Tensor, sigma: float) -> Tensor:
        """Make guided attention mask.
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen.item(), device=olen.device), torch.arange(ilen.item(), device=ilen.device))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(-(grid_y / ilen - grid_x / olen) ** 2 / (2 * (sigma ** 2)))
    
    @staticmethod
    def _make_masks(ilens: Tensor, olens: Tensor) -> Tensor:
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
        """
        in_masks = get_mask(ilens)  # (B, T_in)
        out_masks = get_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)

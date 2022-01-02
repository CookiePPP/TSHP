import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieSpeech.utils.modules.core import nnModule, ResBlock, ConvNorm
from CookieSpeech.utils.modules.local_attention import LocalAttention, SinusoidalEmbeddings, apply_rotary_pos_emb_qk, apply_rotary_pos_emb
from CookieSpeech.utils.modules.utils import get_mask1d


class TransformerEncoderLayer(nnModule):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward:int = 2048,
            ff_kernel_size:int = 1,
            n_blocks = 1,
            n_layers = 2,
            cond_dim = 0,
            dropout: float = 0.1,
            window_size: int = 0,
            rezero: bool = False,
            act_func: str = 'relu',
            separable: bool = False,
            weight_norm: bool = False,
            instance_norm: bool = False,
            layer_norm: bool = False,
            batch_norm: bool = False,):
        super().__init__()
        if window_size is None:
            window_size = 0
        if int(window_size):
            self.self_attn = LocalAttention(window_size, dim=d_model)
        else:
            self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, kdim=d_model, vdim=d_model)
            self.rel_pos = SinusoidalEmbeddings(dim=d_model)
        self.local = bool(int(window_size))
        self.dropout = nn.Dropout(dropout)
        
        if not rezero:
            self.att_norm = nn.LayerNorm(d_model)
        if rezero:
            self.att_res_weight = nn.Parameter(torch.ones(1)*0.01)
        
        self.resblock = ResBlock(
            d_model, d_model, dim_feedforward, kernel_size=ff_kernel_size, n_blocks=n_blocks, n_layers=n_layers,
            cond_dim=cond_dim, rezero=rezero, dropout=dropout, separable=separable, weight_norm=weight_norm,
            instance_norm=instance_norm, layer_norm=layer_norm, batch_norm=batch_norm, act_func=act_func)
    
    def forward(self,
                src,  # [B, in_T, dim]
                cond=None,  # [B, 1 or in_T, cond_dim]
                mask=None,  # [B, in_T,   1]
                ):
        if True:# Multi-Head Attention
            
            if self.local:
                src2 = self.self_attn(src, src, src, mask)
                enc_align = None
            else:
                pos_emb = self.rel_pos(src)
                q, k = apply_rotary_pos_emb_qk(src, src, pos_emb)
                src2 = self.self_attn(
                    q  .transpose(0, 1),
                    k  .transpose(0, 1),
                    src.transpose(0, 1),
                    key_padding_mask= None if mask is None else ~mask.squeeze(2)
                )[0].transpose(0, 1)
            src2 = self.dropout(src2)
            if hasattr(self, 'att_res_weight'):
                src2 = self.att_res_weight*src2
            src = src + src2# [B, in_T, dim]
            if hasattr(self, 'att_norm'):
                src = self.att_norm(src)
        
        src = self.resblock(src, cond, mask=mask)
        
        if mask is not None:
            src = src.masked_fill(~mask, 0.0)
        return src
            #  [B, in_T, dim], [B, in_T, in_T]

class FeedForwardTransformer(nnModule):
    def __init__(self, hidden_dim, n_global_layers, in_dim=None, out_dim=None, cond_dim=0, n_heads=4, dim_feedforward=None, kernel_size=1, dropout=0.1, n_local_layers=0, separable=False, layer_norm=False, weight_norm=False, window_size=None, rezero=True, k_pos_embed=False):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = hidden_dim * 2
        if in_dim is None:
            in_dim = hidden_dim
        if out_dim is None:
            out_dim = hidden_dim
        if n_local_layers and not window_size:
            print("WARNING: using local FFT layers without specified window_size")
        if in_dim != hidden_dim:
            self.pre_conv = ConvNorm(in_dim, hidden_dim)
        self.k_pos_embed = k_pos_embed
        self.enc_layers = nn.ModuleList()
        for _ in range(n_local_layers):
            self.enc_layers.append(
                TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=dim_feedforward, ff_kernel_size=kernel_size, dropout=dropout, window_size=window_size, rezero=rezero, cond_dim=cond_dim, separable=separable, layer_norm=layer_norm, weight_norm=weight_norm)
            )
        for _ in range(n_global_layers):
            self.enc_layers.append(
                TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=dim_feedforward, ff_kernel_size=kernel_size, dropout=dropout, window_size=0, rezero=rezero, cond_dim=cond_dim, separable=separable, layer_norm=layer_norm, weight_norm=weight_norm)
            )
        if hidden_dim != out_dim:
            self.post_conv = ConvNorm(hidden_dim, out_dim)
    
    def forward(self,
            x,        # [B,      in_T,      dim]
            cond=None,# [B, 1 or in_T, cond_dim]
            mask=None,# [B,      in_T,        1]
            lens=None,# [B,         1,        1]
        ):
        if mask is None and lens is not None:
            mask = get_mask1d(lens)
        if hasattr(self, 'pre_conv'):
            x = self.pre_conv(x)
        for i, layer in enumerate(self.enc_layers):
            x = layer(x, cond=cond, mask=mask)
        if self.k_pos_embed:
            pos_emb = self.enc_layers[-1].rel_pos(x)
            x = apply_rotary_pos_emb(x, pos_emb)
        if hasattr(self, 'post_conv'):
            x = self.post_conv(x)
        return x

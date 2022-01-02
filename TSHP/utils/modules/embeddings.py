from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from TSHP.utils.modules.core import nnModule
from math import sqrt


class TextEmbedding(nnModule):
    def __init__(self, n_symbols, text_embed_dim, padding_idx: Optional[int] = 0):
        super(TextEmbedding, self).__init__()
        self.text_embedding = nn.Embedding(n_symbols, text_embed_dim, padding_idx=padding_idx)
        std = sqrt(2 / text_embed_dim)
        nn.init.normal_(self.text_embedding.weight.data, mean=0.0, std=std)
    
    def forward(self, text_ids):  # LongTensor[B, txt_T, 1]
        assert text_ids.dim() == 3, f'text_ids has {text_ids.dim()} dims, expected 3 of shape [B, txt_T, 1]'
        text_embed = self.text_embedding(text_ids.squeeze(2))
        return text_embed  # [B, txt_T, text_embed]


class SpeakerEmbedding(nnModule):
    def __init__(self, h):
        super(SpeakerEmbedding, self).__init__()
        self.out_dim = h.speaker_embed_dim
        self.spkr_embedding = nn.Embedding(h.n_speakers, h.speaker_embed_dim)
        std = sqrt(2 / h.speaker_embed_dim)
        std = std / h.spkr_grad_mul
        nn.init.normal_(self.spkr_embedding.weight.data, mean=0.0, std=std)
        # self.conv = ConvStack(h.spkr_embed_dim+4, h.spkr_embed_dim*2, h.spkr_embed_dim*2, n_layers=3, kernel_size=1, act_func=nn.ReLU(), residual=False, dropout=0.2)
        # self.GLU = GLU(h.spkr_embed_dim*2, h.spkr_embed_dim)
        self.spkr_grad_mul = h.spkr_grad_mul
        self.weight = nn.Parameter(torch.ones(1))
    
    def forward(self, spkr_ids):  # LongTensor[B, 1, 1]
        assert spkr_ids.dim() == 3, f'speaker_ids has {spkr_ids.dim()} dims, expected 3 of shape [B, 1, 1]'
        spkr_embed = self.spkr_embedding(spkr_ids.squeeze(2))
        spkr_embed = (spkr_embed * self.spkr_grad_mul) * self.weight
        return spkr_embed  # [B, 1, embed_dim]

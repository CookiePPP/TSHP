import torch
from torch import nn
from torch.nn import functional as F

from CookieSpeech.utils.modules.utils import get_mask1d, maybe_cat
from CookieSpeech.utils.modules.core import ResBlock, nnModule, reparameterize
from CookieSpeech.utils.modules.loss_func.common import kld_loss

from CookieSpeech.utils.modules.embeddings import TextEmbedding

class TextEncoder(nnModule):
    def __init__(self, n_symbols, text_embed_dim, hidden_dim, out_dim, n_blocks, n_layers, kernel_size, rezero=True, **conv_params):
        super().__init__()
        self.emb = TextEmbedding(n_symbols, text_embed_dim)
        self.enc = ResBlock(
            text_embed_dim,
            out_dim, hidden_dim,
            n_blocks, n_layers,
            kernel_size=kernel_size, rezero=rezero, **conv_params,
        )
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, 1, bias=True, batch_first=True, bidirectional=True,
        )
    
    def forward(self, text_ids, cond=None, txt_mask=None, txt_lens=None):
        text_embed = self.emb(text_ids)
        text_acts  = self.enc(text_embed, cond, txt_mask)
        
        text_acts = nn.utils.rnn.pack_padded_sequence(text_acts, txt_lens.view(-1).cpu().numpy(), batch_first=True, enforce_sorted=False)
        text_acts, (h0, _) = self.lstm(text_acts)
        text_acts, _ = nn.utils.rnn.pad_packed_sequence(text_acts, batch_first=True)# -> [B, txt_T, C]
        text_actsf, text_actsb = text_acts.chunk(2, dim=2)# split forward and backwards outputs
        text_acts = text_actsf + text_actsb # then add them elementwise
        
        if txt_mask is not None:
            text_acts = text_acts.masked_fill_(~txt_mask, 0.0)
        return text_acts


class BertVAE(nnModule):
    def __init__(self,
                 moji_dim: int, bert_dim: int, textenc_dim: int,  hidden_dim: int,
                 n_tokens: int, n_blocks: int,    n_layers: int, kernel_size: int,
                 rezero=True, **conv_params):
        super().__init__()
        self.moji_dim    = moji_dim
        self.bert_dim    = bert_dim
        self.textenc_dim = textenc_dim
        self.enc = ResBlock(
            self.moji_dim+self.bert_dim+textenc_dim,
            n_tokens*2, hidden_dim,
            n_blocks, n_layers,
            kernel_size=kernel_size, rezero=rezero, **conv_params,
        )
        self.dec = ResBlock(
            n_tokens,
            self.bert_dim, hidden_dim,
            n_blocks, n_layers,
            kernel_size=kernel_size, rezero=rezero, **conv_params,
        )
    
    def forward(self, moji_embed, bert_embed, text_acts, txt_mask, cond=None):
        _ = maybe_cat((moji_embed, bert_embed, text_acts), dim=2)# -> [B, txt_T, bert+moji]
        mulogvar = self.enc(_, cond, txt_mask)
        mu, logvar = mulogvar.chunk(2, dim=2)
        z = reparameterize(mu, logvar, self.training)
        kld = kld_loss(mu, logvar)
        bert_acts = self.dec(z, cond, txt_mask)
        return bert_acts, kld

class TextModule(nnModule):
    def __init__(self, mh,
                 n_symbols: int, text_embed_dim: int, out_dim: int, cond_dim,
                 moji_dim: int = 0, bert_dim: int = 0, bert_vae=False, aux_dropout=0.0):
        super().__init__()
        self.mh = mh
        self.cond_dim = cond_dim
        self.textenc = TextEncoder(
            n_symbols,
            text_embed_dim,
            mh.textenc.hidden_dim,
            mh.textenc.out_dim,
            mh.textenc.n_blocks,
            mh.textenc.n_layers,
            mh.textenc.kernel_size,
            mh.textenc.rezero,
            **{**mh.textenc.conv_params, 'cond_dim': cond_dim},
        )
        
        self.aux_dropout = aux_dropout
        self.moji_dim = moji_dim or getattr(mh, 'moji_dim', 0)
        self.bert_dim = bert_dim or getattr(mh, 'bert_dim', 0)
        if bert_vae:
            self.bertvae = BertVAE(
                self.moji_dim,
                self.bert_dim,
                mh.textenc.out_dim,
                mh.bertvae.hidden_dim,
                mh.bertvae.n_tokens,
                mh.bertvae.n_blocks,
                mh.bertvae.n_layers,
                mh.bertvae.kernel_size,
                mh.bertvae.rezero,
                **{**mh.bertvae.conv_params, 'cond_dim': cond_dim},
            )
        
        self.mem_input_dim = mh.textenc.out_dim + self.bert_dim + self.moji_dim + cond_dim
        self.memenc = ResBlock(
            self.mem_input_dim,
            out_dim or mh.memenc.out_dim,
            mh.memenc.hidden_dim,
            mh.memenc.n_blocks,
            mh.memenc.n_layers,
            mh.memenc.kernel_size,
            mh.memenc.rezero,
            **{**mh.memenc.conv_params, 'cond_dim': cond_dim},
        )
    
    def forward(self, text_ids, moji_embed, bert_embed, txt_lens, cond=None):
        txt_mask = get_mask1d(txt_lens)
        text_acts = self.textenc(text_ids, cond, txt_mask, txt_lens)
        if hasattr(self, 'bertvae'):
            bert_acts, kld = self.bertvae(moji_embed, bert_embed, text_acts, txt_mask, cond=cond)
        else:
            bert_acts = bert_embed
            kld = text_acts[:, :, :1]*0.0
        if self.aux_dropout:
            bert_acts, moji_embed = F.dropout(bert_acts, 0.5), F.dropout(moji_embed, 0.5)
        mem_input = maybe_cat((text_acts, bert_acts, moji_embed, cond if self.cond_dim else None), dim=2)
        text_acts = self.memenc(mem_input, cond, txt_mask)
        return text_acts, kld

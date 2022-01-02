import torch
import torch.nn as nn
import torch.nn.functional as F

from CookieSpeech.utils.modules.core import ConvNorm, ResBlock
from CookieSpeech.utils.modules.rnn import LSTMBlock
from CookieSpeech.utils.modules.decoders.rnn_att.hybrid import Attention


class RecurrentBlock(nn.Module):
    def __init__(self, rb, input_dim, cond_dim=0, randn_dim=0, proj_logvar=False, output_dim=None, act_func='relu'):
        super(RecurrentBlock, self).__init__()
        self.input_dim = input_dim
        self.randn_dim = randn_dim
        self.n_frames_per_step = rb.n_frames_per_step
        self.attlstm_n_states = rb.attlstm_n_layers * 2
        self.declstm_n_states = rb.declstm_n_layers * 2
        self.attention_n_states = 2
        self.att_value_dim = rb.att_value_dim
        self.attlstm_dim   = rb.attlstm_dim
        self.declstm_dim   = rb.declstm_dim
        
        # modules
        self.prenet = ResBlock(
            input_dim * self.n_frames_per_step, rb.prenet_dim, rb.prenet_dim,
            1, rb.prenet_n_layers, kernel_size=1, rezero=False, skip_all_res=True,
            dropout=rb.prenet_dropout, always_dropout=rb.prenet_always_dropout,
            cond_dim=cond_dim, act_func=act_func, LSUV_init=False, LSUV_init_bias=True)
        
        input_dim = rb.prenet_dim + rb.att_value_dim
        if self.randn_dim:
            self.randn_lin = ConvNorm(self.randn_dim, rb.prenet_dim)
            if getattr(rb, 'randn_rezero', False):
                self.randn_rezero = nn.Parameter(torch.tensor(0.0))
        self.attLSTM   = LSTMBlock(input_dim, rb.attlstm_dim, n_layers=rb.attlstm_n_layers, dropout=0.0, zoneout=rb.attlstm_zoneout, residual=True)
        self.attention = Attention(rb, cond_dim=cond_dim, bias=getattr(rb, 'att_bias', True))
        input_dim = rb.attlstm_dim + rb.att_value_dim
        self.decLSTM   = LSTMBlock(input_dim, rb.declstm_dim, n_layers=rb.declstm_n_layers, dropout=0.0, zoneout=rb.declstm_zoneout, residual=True)
        
        self.projnet = ResBlock(
            rb.att_value_dim + rb.declstm_dim,
            output_dim*(1+proj_logvar)*self.n_frames_per_step,
            rb.att_value_dim + rb.declstm_dim,
            n_blocks=1, n_layers=1, kernel_size=1,
            skip_all_res=True, rezero=False,
            cond_dim=cond_dim, act_func=act_func)
    
    def update_kv(self, encoder_outputs, text_lengths, cond):
        self.attention.update_kv(encoder_outputs, text_lengths, cond)
    
    def reset_kv(self):
        self.attention.reset_kv()
    
    def pre(self, x, cond=None, mask=None, reshape_input=True):# [B, T, in_dim]
        B, T, in_dim = x.shape
        if reshape_input:
            x = x.reshape(B, T//self.n_frames_per_step, in_dim*self.n_frames_per_step)# [B, T/nfps, in_dim*nfps]
            if mask is not None and self.n_frames_per_step > 1:
                mask = F.avg_pool1d(mask.transpose(1, 2).float(), kernel_size=self.n_frames_per_step).transpose(1, 2).round().bool()
        
        x = self.prenet(x, cond, mask)
        if self.randn_dim:
            rx = torch.randn(*x.shape[:-1], self.randn_dim, device=x.device, dtype=x.dtype)
            if self.sigma is not None:
                rx *= self.sigma
            if self.randn_max is not None:
                rx = rx.clamp(min=-self.randn_max, max=self.randn_max)
            rx = self.randn_lin(rx)
            if hasattr(self, 'randn_rezero'):
                rx = self.randn_rezero*rx
            x = x + rx
        return x# [B, T, H]
    
    def post(self, x, v, cond=None, mask=None, reset_kv=False, reshape_output=True):# [B, T, declstm_dim], [B, T, vdim]
        if type(x) in (list, tuple):
            x = torch.stack(x, dim=1)# [[B, declstm_dim],]*(T//n_fps) -> [B, T//n_fps, declstm_dim]
        if type(v) in (list, tuple):
            v = torch.stack(v, dim=1)# [[B, vdim],]*(T//n_fps) -> [B, T//n_fps, vdim]
        if mask is not None and self.n_frames_per_step > 1:
            mask = F.avg_pool1d(mask.transpose(1, 2).float(), kernel_size=self.n_frames_per_step).transpose(1, 2).round().bool()
        
        x = torch.cat((x, v), dim=2)# [B, T//n_fps, declstm_dim], [B, T//n_fps, vdim] -> [B, T//n_fps, declstm_dim+vdim]
        x = self.projnet(x, cond, mask)# -> [B, T//n_fps, output_dim*n_fps]
        if reshape_output:
            x = x.view(x.shape[0], x.shape[1]*self.n_frames_per_step, x.shape[2]//self.n_frames_per_step)# -> [B, T, output_dim]
        if reset_kv:
            self.reset_kv()
        return x# [B, T, output_dim]
    
    def get_attlstm_input(self, x, v):
        if v is None:
            return torch.cat((x, torch.zeros(x.shape[0], self.att_value_dim, device=x.device, dtype=x.dtype)), dim=1)
        else:
            return torch.cat((x, v), dim=1)
    
    # [B, prenet_dim], [B, vdim], List[Tensor], List[Tensor], List[Tensor]
    def main(self, x, cv, attlstm_states, att_states, declstm_states, cond=None):
        
        x = self.get_attlstm_input(x, cv)
        x, attlstm_states = self.attLSTM(x, attlstm_states)
        
        v, alignment, att_states = self.attention(x, att_states, cond)
        # print(x.shape, v.shape)# torch.Size([4, 512]) torch.Size([4, 1, 256])
        x, declstm_states = self.decLSTM(torch.cat((x, v.squeeze(1)), dim=1), declstm_states)
        return x, v, alignment, attlstm_states, att_states, declstm_states
    # [B, C], [B, vdim], [B, 1, txt_T], List[Tensor], List[Tensor], List[Tensor]
    
    def forward(self, x, v, states, cond=None):# ff_forward # [B, 1, prenet_dim], [B, 1, vdim], List[List[Tensor]]
        if states is None:
            states = [None]*3
        x, v, alignment, *states = self.main(x, v, *states, cond=cond)
        return x, v, alignment, states
        # [B, C], [B, vdim], [B, 1, txt_T], List[List[Tensor]]
    
    def ar_forward(self, x, v, states, cond):# [B, 1, in_dim*n_fps], [B, 1, vdim], List[List[Tensor]]
        if states is None:
            states = [None]*3
        x = self.pre(x.unsqueeze(1), cond, reshape_input=False).squeeze(1)
        x, v, alignment, *states = self.main(x, v, *states, cond=cond)
        x = self.post(x.unsqueeze(1), v.unsqueeze(1), cond, reshape_output=True)
        return x, v, alignment, states
        # [B, n_fps, output_dim], [B, vdim], [B, 1, txt_T], List[List[Tensor]]

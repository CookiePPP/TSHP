import torch
import torch.nn as nn
import torch.nn.functional as F
from CookieSpeech.utils.modules.utils import get_mask1d
from CookieSpeech.utils.modules.core import ConvNorm, CondConv


class LocationLayer(nn.Module):
    def __init__(self, n_filters, kernel_size,
                 att_dim, out_bias=False, att_loc_act_func='relu', act_func_bias=False, cond_dim=0, w_gain=0.1):
        super(LocationLayer, self).__init__()
        self.location_conv  = CondConv(        2, n_filters, kernel_size=kernel_size, bias=act_func_bias, cond_dim=cond_dim,
                                               act_func=att_loc_act_func, w_gain=w_gain)
        self.location_dense = CondConv(n_filters,   att_dim, bias=out_bias, w_init_gain='tanh', cond_dim=cond_dim)
    
    def forward(self, attention_weights_cat, cond):# [B, txt_T, 2]
        processed_attention = self.location_conv(attention_weights_cat, cond)# [B, txt_T, 2] -> [B, txt_T, n_filters]
        processed_attention = self.location_dense(processed_attention, cond) # -> [B, txt_T, att_dim]
        return processed_attention# [B, txt_T, att_dim]

class Attention(nn.Module):
    def __init__(self, h, cond_dim=0, bias=True):
        """
        Standard Attention Module
        
        usage:
        >>> attention = Attention(h)
        >>> attention.update_kv(memory, txt_lens) # add key tensor to module
        >>> for frame in frames:
        >>>     ... # get query somewhere from input
        >>>     v, alignment, states = attention(query, states) # use query + key to produce value
        >>>     ... # use v for whatever
        >>> attention.reset_kv()
        """
        super().__init__()
        self.random_energy = getattr(h, 'random_energy', False)
        
        self.key_encoder    = CondConv(h.att_value_dim, h.att_dim, bias=bias, cond_dim=cond_dim, w_gain=h.contentk_w_gain, w_init_gain='tanh')
        self.query_encoder  = ConvNorm(h.  attlstm_dim, h.att_dim, bias=bias, w_gain=h.contentq_w_gain, w_init_gain='tanh')
        self.location_layer = LocationLayer(32, 31, h.att_dim, out_bias=bias, att_loc_act_func=h.att_loc_act_func, act_func_bias=h.act_func_bias, w_gain=h.location_w_gain)
        self.v = ConvNorm(h.att_dim, 1, bias=False, w_gain=1.0)
        
        self.window_offset = h.att_window_offset
        self.window_range  = h.att_window_range
    
    def update_kv(self, value, value_lengths, cond):# [B, txt_T, v_dim], LongTensor[B]
        self.value = value                        # -> [B, txt_T, v_dim]
        self.key   = self.key_encoder(value, cond)# -> [B, txt_T, key_dim]
        self.value_lengths = value_lengths        # -> LongTensor[B, 1, 1]
        self.value_mask = get_mask1d(value_lengths)# -> BoolTensor[B, txt_T, 1]
    
    def reset_kv(self):
        if hasattr(self, 'key'):
            del self.key
        if hasattr(self, 'value'):
            del self.value
        if hasattr(self, 'value_lengths'):
            del self.value_lengths
        if hasattr(self, 'value_mask'):
            del self.value_mask
    
    def sep_states(self, states):
        attention_weights_cat, current_pos = states
        return attention_weights_cat, current_pos# [B, txt_T, 2], [B]
    
    def init_states(self, x):
        assert hasattr(self, 'value'), '.update_kv() must be called before .init_states()'
        batch_size = x.shape[0]
        attention_weights_cat = torch.zeros(batch_size, self.value_lengths.max().item(), 2, device=x.device, dtype=x.dtype)
        attention_weights_cat[:, 0].fill_(1.0)
        current_pos = torch.ones(batch_size, device=x.device, dtype=x.dtype)
        return attention_weights_cat, current_pos# [B, txt_T, 2], [B]
    
    def col_states(self, attention_weights_cat, current_pos):
        states = [attention_weights_cat, current_pos,]
        return states# List[FloatTensor]
    
    def forward(self, query, states, cond=None):# [B, qdim], List[FloatTensor]
        if states is None:
            attention_weights_cat, current_pos = self.init_states(query)
        else:
            attention_weights_cat, current_pos = self.sep_states(states)
        
        processed = self.location_layer(attention_weights_cat, cond)# [B, txt_T, 2] -> [B, txt_T, att_dim]
        processed.add_( self.query_encoder(query.unsqueeze(1)).expand_as(self.key) )# -> [B, 1, att_dim] -> [B, txt_T, att_dim]
        processed.add_( self.key )# -> [B, txt_T, att_dim]
        alignment = self.v( torch.tanh( processed ) )# [B, txt_T, 1]
        
        mask = ~self.value_mask# [B, txt_T, 1]
        B, txt_T, _ = attention_weights_cat.shape
        if self.window_range > 0 and current_pos is not None:
            current_pos = current_pos.view(-1)
            if self.window_offset:
                current_pos = current_pos + self.window_offset
            max_end = self.value_lengths.view(-1) -1 -self.window_range
            min_start = self.window_range
            current_pos = torch.min(current_pos.clamp(min=min_start), max_end.to(current_pos))
            
            mask_start = (current_pos-self.window_range).clamp(min=0).round().unsqueeze(1).expand(-1, txt_T).unsqueeze(2)# [B, txt_T, 1]
            mask_end = (mask_start+(self.window_range*2)).round()  # [B, txt_T, 1]
            pos_mask = torch.arange(txt_T, device=current_pos.device).expand(B, -1).unsqueeze(-1)# [B, txt_T, 1]
            pos_mask = (pos_mask >= mask_start) & (pos_mask <= mask_end)# [B, txt_T, 1]
        
            # attention_weights_cat[pos_mask].view(B, self.window_range*2+1) # for inference masked_select later
            mask = mask | ~pos_mask# [B, txt_T, 1] & [B, txt_T, 1] -> [B, txt_T, 1]
        alignment.data.masked_fill_(mask, -float('inf'))# [B, txt_T, 1]
    
        if self.random_energy and self.training:
            alignment += torch.randn(alignment.shape[0], dtype=alignment.dtype, device=alignment.device).mul(self.random_energy)
        
        alignment = F.softmax(alignment, dim=1)# [B, txt_T, 1] # softmax along encoder tokens dim
        
        attention_context = (alignment.transpose(1, 2) @ self.value).squeeze(1)
        #              [B, 1, txt_T] @ [B, txt_T, enc_dim] -> [B, 1, enc_dim]
        
        new_pos = (alignment.detach()*torch.arange(txt_T, device=alignment.device).expand(B, -1).unsqueeze(2)).sum(1, keepdim=True)
        # ([B, txt_T, 1] * [B, txt_T, 1]).sum(1) -> [B, 1, 1]
        
        new_attention_weights_cat = torch.cat((alignment, attention_weights_cat[:, :, 1:2]+alignment), dim=2)
        #             cat([B, txt_T, 1], [B, txt_T, 1]) -> [B, txt_T, 2]
        new_states = self.col_states(new_attention_weights_cat, new_pos)
        return attention_context, alignment, new_states# [B, v_dim], [B, txt_T, 1], List[FloatTensor]

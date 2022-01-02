import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch.nn.parameter import Parameter

from CookieSpeech.utils.modules.core import nnModule, ResBlock
from torch import Tensor
from typing import List, Tuple, Optional, Union, Any
from torch.nn.modules.rnn import LSTMCell

#class LSTMCell(jit.ScriptModule):
#    def __init__(self, input_size: int, hidden_size: int, bias:bool):
#        super(LSTMCell, self).__init__()
#        self.input_size = input_size
#        self.hidden_size = hidden_size
#        self.bias = bias
#        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
#        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
#        nn.init.xavier_uniform_(self.weight_ih.data, gain=1.0)
#        nn.init.xavier_uniform_(self.weight_hh.data, gain=1.0)
#        if bias:
#            self.bias_ih = Parameter(torch.zeros(4 * hidden_size))
#            self.bias_hh = Parameter(torch.zeros(4 * hidden_size))
#    
#    @jit.script_method
#    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
#        hx, cx = state
#        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
#                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
#        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
#        
#        ingate = torch.sigmoid(ingate)
#        forgetgate = torch.sigmoid(forgetgate)
#        cellgate = torch.tanh(cellgate)
#        outgate = torch.sigmoid(outgate)
#
#        cy = (forgetgate * cx) + (ingate * cellgate)
#        hy = outgate * torch.tanh(cy)
#
#        return hy, (hy, cy)

class LSTMCellWithZoneout(nn.LSTMCell):# taken from https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/tacotron2/decoder.html
                                           # and modified with dropout and to use LSTMCell as base
    """ZoneOut Cell module.
    
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.

    Examples:
        >>> lstm = LSTMCellWithZoneout(16, 32, zoneout=0.5)

    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305

    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch

    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0, zoneout: float=0.0, inplace=True):
        super().__init__(input_size, hidden_size, bias)
        self.zoneout_rate = zoneout
        self.dropout_rate = dropout
        self.inplace      = inplace
        if zoneout > 1.0 or zoneout < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError("dropout probability must be in the range from 0.0 to 1.0.")
    
    def forward(self, inputs: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (tuple):
                - Tensor: Batch of initial hidden states (B, hidden_size).
                - Tensor: Batch of initial cell states (B, hidden_size).

        Returns:
            tuple:
                - Tensor: Batch of next hidden states (B, hidden_size).
                - Tensor: Batch of next cell states (B, hidden_size).

        """
        next_hidden = super(LSTMCellWithZoneout, self).forward(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        next_hidden = self._dropout(hidden, next_hidden, self.dropout_rate)
        return next_hidden
    
    def _dropout(self,
                 h     : Union[Tuple[Tensor, Tensor], Tensor],
                 next_h: Union[Tuple[Tensor, Tensor], Tensor], prob: float) -> Union[Tuple[Tensor, Tensor], Tensor, Any]:
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._dropout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )
        
        return F.dropout(next_h, prob, self.training, self.inplace)
    
    def _zoneout(self,
                 h     : Union[Tuple[Tensor, Tensor], Tensor],
                 next_h: Union[Tuple[Tensor, Tensor], Tensor], prob: float) -> Union[Tuple[Tensor, Tensor], Tensor, Any]:
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training and prob > 0.0:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h
                                               
class LSTMBlock(nnModule):# LSTM with variable number of layers, zoneout and residual connections
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.0, zoneout=0.0, residual=False, prenet_n_blocks=0, prenet_n_layers=0, postnet_n_blocks=0, postnet_n_layers=0, net_act_func='relu', out_dim: int = None):
        super(LSTMBlock, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout  = dropout
        self.zoneout  = zoneout
        self.residual = residual
        
        inp_dim = input_dim
        
        if prenet_n_layers > 0 or prenet_n_blocks > 0:
            self.prenet = ResBlock(inp_dim, hidden_dim, hidden_dim, prenet_n_blocks, prenet_n_layers, act_func=net_act_func)
            inp_dim = hidden_dim
        
        self.lstm_cell = nn.ModuleList()
        for i in range(self.n_layers):
            self.lstm_cell.append(LSTMCellWithZoneout(inp_dim, self.hidden_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.hidden_dim
        
        if (inp_dim != out_dim and out_dim is not None) or postnet_n_layers > 0 or postnet_n_blocks > 0:
            self.postnet = ResBlock(inp_dim, out_dim, hidden_dim, postnet_n_blocks, postnet_n_layers, act_func=net_act_func)
            inp_dim = out_dim
    
    def maybe_init_states(self, x: Tensor, states: Optional[Tuple[Tensor, ...]] = None) -> Tuple[Tensor, ...]:
        B = x.shape[0]
        if states is None:
            states = [x[0].new_zeros(B, self.hidden_dim) for i in range(2*self.n_layers)]
        return states
    
    def teacher_force(self, inputs: Tensor, states: Optional[Tuple[Tensor, ...]] = None) -> Tuple[Tensor, ...]:# [B, T, C], TensorList()
        if hasattr(self, 'prenet'):
            inputs = self.prenet(inputs)
        inputs = inputs.unbind(1)# [[B, C], ...]
        outputs = []
        for inpu in inputs:
            output, states = self(inpu, states)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)# [B, T, C]
        if hasattr(self, 'postnet'):
            outputs = self.postnet(outputs)
        return outputs, states# [B, T, C], TensorList()
    
    def forward(self, x, states: Optional[Tuple[Tensor, ...]] = None) -> Tuple[Tensor, Tuple[Tensor, ...]]:# [B, input_dim], List[FloatTensor]
        states = self.maybe_init_states(x, states)
        
        if hasattr(self, 'prenet'):
            x = self.prenet(x.unsqueeze(1)).squeeze(1)
        
        xi = x
        for j, lstm_cell in enumerate(self.lstm_cell):
            states[j*2:j*2+2] = lstm_cell(xi, tuple(states[j*2:j*2+2]))
            xi = xi + states[j*2] if self.residual and j>0 else states[j*2]
        
        if hasattr(self, 'postnet'):
            xi = self.postnet(xi.unsqueeze(1)).squeeze(1)
        
        return xi, states# [B, hidden_dim], List[FloatTensor]



#class GRUCellWithZoneout(nn.Module):
class GRUCellWithZoneout(nn.GRUCell):# taken from https://espnet.github.io/espnet/_modules/espnet/nets/pytorch_backend/tacotron2/decoder.html
                                           # and modified with dropout and to use LSTMCell as base
    """ZoneOut Cell module.
    
    This is a module of zoneout described in
    `Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`_.
    This code is modified from `eladhoffer/seq2seq.pytorch`_.

    Examples:
        >>> gru = GRUCellWithZoneout(16, 32, zoneout=0.5)

    .. _`Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations`:
        https://arxiv.org/abs/1606.01305

    .. _`eladhoffer/seq2seq.pytorch`:
        https://github.com/eladhoffer/seq2seq.pytorch

    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool=True, dropout: float=0.0, zoneout: float=0.0, inplace=True):
        super().__init__(input_size, hidden_size, bias)
        self.zoneout_rate = zoneout
        self.dropout_rate = dropout
        self.inplace      = inplace
        if zoneout > 1.0 or zoneout < 0.0:
            raise ValueError("zoneout probability must be in the range from 0.0 to 1.0.")
        if dropout > 1.0 or dropout < 0.0:
            raise ValueError("dropout probability must be in the range from 0.0 to 1.0.")
    
    def forward(self, inputs: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """Calculate forward propagation.
        
        Args:
            inputs (Tensor): Batch of input tensor (B, input_size).
            hidden (Tensor): Batch of initial hidden states (B, hidden_size).
        
        Returns:
            Tensor: Batch of next hidden states (B, hidden_size).
        
        """
        next_hidden = super(GRUCellWithZoneout, self).forward(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout_rate)
        next_hidden = self._dropout(hidden, next_hidden, self.dropout_rate)
        return next_hidden
    
    def _dropout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._dropout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )
        
        return F.dropout(next_h, prob, self.training, self.inplace)
    
    def _zoneout(self, h, next_h, prob):
        # apply recursively
        if isinstance(h, tuple):
            num_h = len(h)
            if not isinstance(prob, tuple):
                prob = tuple([prob] * num_h)
            return tuple(
                [self._zoneout(h[i], next_h[i], prob[i]) for i in range(num_h)]
            )

        if self.training and prob > 0.0:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h

#class GRUBlock(nn.Module):
class GRUBlock(nnModule):# LSTM with variable number of layers, zoneout and residual connections
    def __init__(self, input_dim, hidden_dim, n_layers=1, dropout=0.0, zoneout=0.1, residual=False):
        super(GRUBlock, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout  = dropout
        self.zoneout  = zoneout
        self.residual = residual
        
        inp_dim = input_dim
        self.gru_cell = []
        for i in range(self.n_layers):
            self.gru_cell.append(GRUCellWithZoneout(inp_dim, self.hidden_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.hidden_dim
        self.gru_cell = nn.ModuleList(self.gru_cell)
    
    def maybe_init_states(self, x, states=None):
        B = x.shape[0]
        if states is None:
            states = [x[0].new_zeros(B, self.hidden_dim) for i in range(self.n_layers)]
        return states
    
    def teacher_force(self, inputs, states=None):# [B, T, C], TensorList()
        inputs = inputs.unbind(1)# [[B, C], ...]
        outputs = []
        for input in inputs:
            output, states = self(input, states)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)# [B, T, C]
        return outputs, states# [B, T, C], TensorList()
    
    def forward(self, x, states=None):# [..., input_dim], TensorList()
        states = self.maybe_init_states(x, states)
        
        xi = x
        for j, gru_cell in enumerate(self.gru_cell):
            states[j] = gru_cell(xi, states[j])
            xi = xi + states[j] if self.residual and j>0 else states[j]
        
        return xi, states# [..., hidden_dim], TensorList()
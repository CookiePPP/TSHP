import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from CookieSpeech.utils.modules.rnn import LSTMCellWithZoneout

from CookieSpeech.utils.modules.core import nnModule, ConvNorm, reparameterize
from torch import Tensor
from typing import List, Tuple, Optional


#@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):# uses mean and standard deviation
    kld = -0.5 +logsigma2 -logsigma1 + 0.5 * (logsigma1.exp().pow(2) + F.mse_loss(mu1, mu2, reduction='none')) / (logsigma2.exp().pow(2))
    return kld

#@torch.jit.script
def gaussian_analytical_kl_var(mu1, mu2, logvar1, logvar2, normal_weight=0.00):# uses mean and variance
    """KLD between 2 sets of guassians. Returns Tensor with same shape as input."""
    neg_logvar2 = -logvar2
    kld = 0.5 * (-1.0 +logvar2 -logvar1 +logvar1.add(neg_logvar2).float().exp() + F.mse_loss(mu1, mu2, reduction='none').clamp(min=1e-6).log().add(neg_logvar2).float().exp() )
    if normal_weight is not None and normal_weight > 0.0:# optional apply loss mu or logvar that differs extemely from normal dist
        kld = kld + (normal_weight * -0.5) * (-mu1.pow(2) +(1.0 +logvar1 -logvar1.exp()))
    return kld

class ResBlock(nnModule):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=True, rezero=True, kernel_sizes: Optional[List[int]] = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3, 3, 1]
        self.down_rate = down_rate
        self.residual = residual
        self.rezero = rezero
        self.conv = []
        for i, kernel_size in enumerate(kernel_sizes):
            inp_dim = in_width  if   i==0                 else middle_width
            out_dim = out_width if 1+i==len(kernel_sizes) else middle_width
            self.conv.append(ConvNorm(inp_dim, out_dim, kernel_size))
        self.conv = nn.ModuleList(self.conv)
        if self.residual and self.rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, x):# [B, C_in, T]
        res = x
        for conv in self.conv:
            x = conv(F.gelu(x))
        if hasattr(self, 'res_weight'):
            if self.res_weight.abs() < 1e-6:
                self.res_weight.data.fill_(self.res_weight.sign().item()*0.03 or 0.03)
            x = x * self.res_weight
        if self.residual:# make res_channels match x_channels then add the Tensors.
            x_channels = x.shape[1]
            B, res_channels, T = res.shape
            if res_channels > x_channels:
                res = res[:, :x_channels, :]
            elif res_channels < x_channels:
                res = torch.cat((res, res.new_zeros(B, x_channels-res_channels, T)), dim=1)
            x = res + x
        if self.down_rate is not None:
            x = F.avg_pool2d(x, kernel_size=self.down_rate)
        return x# [B, C_out, T//down_rate]

class TopDownBlock(nnModule):
    def __init__(self, hparams, hdn_dim, btl_dim, latent_dim, mem_dim=0):
        super().__init__()
        self.use_z_variance = getattr(hparams, 'use_z_variance', False)
        self.std = 0.5
        
        self.mem_dim = mem_dim
        self.btl_dim = btl_dim
        self.latent_dim = latent_dim
        self.enc    = ResBlock(2*btl_dim+mem_dim, hdn_dim,         2*latent_dim, residual=False)# get Z from target + input
        self.prior  = ResBlock(1*btl_dim+mem_dim, hdn_dim, btl_dim+2*latent_dim, residual=False)# guess Z from just input
        self.prior_weight = nn.Parameter(torch.ones(1)*0.01)
        self.z_proj = nn.Conv1d(latent_dim, btl_dim, 1)# expand Z to the input dim
        if getattr(hparams, 'topdown_resnet_enable', True):
            self.resnet = ResBlock(btl_dim, hdn_dim, btl_dim, residual=True)
    
    def get_z(self, x, z_acts, attention_contexts):
        enc_input = [x, z_acts]
        if self.mem_dim:
            enc_input.append(attention_contexts)
        enc_input = torch.cat(enc_input, dim=1)
        z_mu, z_logsigma = self.enc(enc_input).chunk(2, dim=1)
        return z_mu, z_logsigma
    
    def pred_z(self, x, attention_contexts):
        res = x
        if self.mem_dim:
            x = torch.cat((x, attention_contexts), dim=1)
        x = self.prior(x)
        x, zp_mu, zp_logsigma = x.split([self.btl_dim, self.latent_dim, self.latent_dim], dim=1)
        x = res + x*self.prior_weight
        return x, zp_mu, zp_logsigma
    
    def get_z_embed(self, z_mu, z_logsigma):
        z = reparameterize(z_mu, z_logsigma, self.training)
        z_embed = self.z_proj(z)
        return z_embed
    
    def forward(self,  x,#      x:'Comes from previous DecBlock'                     FloatTensor[B, btl_dim, T]
                  z_acts,# z_acts:'Comes from spect ResBlocks during training'       FloatTensor[B, btl_dim, T]
 attention_contexts=None,#attent:'Comes from the Text Encoder after being expanded' FloatTensor[B, mem_dim, T]
           mel_mask=None,
         use_pred_z=False):
        z_mu, z_logsigma = self.get_z(x, z_acts, attention_contexts)
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)
        
        z_embed = self.get_z_embed(z_mu, z_logsigma)
        if use_pred_z:
            z_embed = self.get_z_embed(z_mu*0.01+zp_mu*0.99, z_logsigma*0.01+zp_logsigma*0.99)# -> [B, latent_dim]
        else:
            z_embed = self.get_z_embed(z_mu, z_logsigma)# -> [B, latent_dim] 
        x = x + z_embed
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
            if mel_mask is not None:
                x.masked_fill_(~mel_mask, 0.0)
        
        if self.use_z_variance:
            kl = gaussian_analytical_kl_var(z_mu, zp_mu, z_logsigma, zp_logsigma)# [B, latent_dim, T]
        else:
            kl = gaussian_analytical_kl(z_mu, zp_mu, z_logsigma, zp_logsigma)# [B, latent_dim, T]
        return x, kl# [B, btl_dim, T], [B, latent_dim, T]
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)
        
        z_embed = self.get_z_embed(zp_mu, zp_logsigma)
        x = x + z_embed
        x.masked_fill_(~mel_mask, 0.0)
        
        if hasattr(self, 'resnet'):
            x = self.resnet(x)
            x.masked_fill_(~mel_mask, 0.0)
        return x

class LSTMBlock(nnModule):
    def __init__(self, in_width, middle_width, out_seq_width, out_vec_width, down_rate=None, residual=False, rezero=True, output_sequence=True, output_vector=True, lstm_n_layers=1):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.rezero = rezero
        assert output_sequence or output_vector, 'this LSTM block has no outputs!'
        if self.residual:
            assert in_width==out_seq_width, 'residual is True but in_width != out_vec_width'
        self.output_sequence = output_sequence
        self.output_vector = output_vector
        
        self.conv_pre = nn.Conv1d(in_width,  middle_width, 1)
        self.lstm = nn.LSTM(middle_width, middle_width, lstm_n_layers, batch_first=True, dropout=0.0, bidirectional=True)
        if self.output_vector:
            self.lstm_proj = nn.Linear(2*middle_width*lstm_n_layers, out_vec_width)
        if self.output_sequence:
            self.conv_post = nn.Conv1d(2*middle_width, out_seq_width, 1)
        if self.residual and self.rezero:
            self.res_weight = nn.Parameter(torch.ones(1)*0.01)
    
    def forward(self, x):# [B, in_width, T]
        B, in_width, T = x.shape
        output = []
        
        res = x
        x = self.conv_pre(F.gelu(x))# -> [B, middle_width, T]
        x, (h, c) = self.lstm(x.transpose(1, 2))
        if self.output_sequence:
            x = x.transpose(1, 2)# -> [B, middle_width, T]
            x = self.conv_post(F.gelu(x))# -> [B, out_width, T]
            if hasattr(self, 'res_weight'):
                if self.res_weight.abs() < 1e-6:
                    self.res_weight.data.fill_(self.res_weight.sign().item()*0.03 or 0.03)
                x = x*self.res_weight
            if self.residual:
                x = res + x
            if self.down_rate is not None:
                x = F.avg_pool2d(x, kernel_size=self.down_rate)
            output.append(x)
        
        if self.output_vector:
            h = h.view(self.lstm.num_layers*2, B, self.lstm.hidden_size).transpose(0, 1).reshape(B, -1)# -> [B, n_layers*2*lstm_dim]
            h = self.lstm_proj(h)# -> [B, out_width]
            output.append(h)
        return tuple(output) if len(output) > 1 else output[0]

class TopDownLSTMBlock(nnModule):
    def __init__(self, hparams, inp_dim, hdn_dim, btl_dim, latent_dim, mem_dim=0, n_layers=2):
        super().__init__()
        self.use_z_variance = getattr(hparams, 'use_z_variance', False)
        self.std = 0.5
        
        assert mem_dim, 'TopDownLSTMBlock requires mem_dim and attention_contexts'
        self.inp_dim = inp_dim
        self.mem_dim = mem_dim
        self.btl_dim = btl_dim
        self.latent_dim = latent_dim
        
        if hparams.exp_cond_proj:
            self.cond_proj = nn.Conv1d(mem_dim, mem_dim, 1)
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        
        self.enc    = LSTMBlock(inp_dim+mem_dim+btl_dim, hdn_dim, out_seq_width=   None, out_vec_width=2*latent_dim, residual=False, output_sequence=False, output_vector=True, lstm_n_layers=n_layers)# get global Z from global z_acts + local input
        self.prior  = LSTMBlock(inp_dim+mem_dim        , hdn_dim, out_seq_width=btl_dim, out_vec_width=2*latent_dim, residual=False, output_sequence= True, output_vector=True, lstm_n_layers=n_layers)# guess global Z from just local input
        if self.inp_dim:
            self.prior_weight = nn.Parameter(torch.ones(1)*0.01)
        self.z_proj = nn.Linear(latent_dim, btl_dim, 1)# expand Z to the input dim
        nn.init.zeros_(self.z_proj.weight)
        nn.init.zeros_(self.z_proj.bias)
        self.resnet = ResBlock(btl_dim, hdn_dim, btl_dim, residual=True)
    
    def get_z(self, x, z_acts, attention_contexts):
        assert attention_contexts is not None and z_acts is not None
        enc_input = [z_acts,]
        if self.inp_dim:
            assert x is not None
            enc_input.append(x)
        if self.mem_dim:
            enc_input.append(attention_contexts)
        enc_input = torch.cat(enc_input, dim=1) if len(enc_input) > 1 else enc_input[0]
        z_mu, z_logsigma = self.enc(enc_input).chunk(2, dim=1)
        return z_mu, z_logsigma
    
    def pred_z(self, x=None, attention_contexts=None):
        assert attention_contexts is not None
        if self.inp_dim:
            assert x is not None
            res = x
            attention_contexts = torch.cat((attention_contexts, x), dim=1)
        x, zp_mulogsigma = self.prior(attention_contexts)
        zp_mu, zp_logsigma = zp_mulogsigma.chunk(2, dim=1)
        if self.inp_dim:
            x = res + x*self.prior_weight
        return x, zp_mu, zp_logsigma
    
    def get_z_embed(self, z_mu, z_logsigma):
        z = reparameterize(z_mu, z_logsigma, self.training)# -> [B, latent_dim]
        z_embed = self.z_proj(z)# -> [B, embed]
        return z_embed
    
    def forward(self, x=None,#      x:'Comes from previous DecBlock'                     FloatTensor[B, btl_dim, T]
                 z_acts=None,# z_acts:'Comes from spect ResBlocks during training'       FloatTensor[B, btl_dim, T]
     attention_contexts=None,#attent:'Comes from the Text Encoder after being expanded' FloatTensor[B, mem_dim, T]
               mel_mask=None,
             use_pred_z=False,):
        assert z_acts is not None, 'z_acts is None'
        assert attention_contexts is not None, 'attention_contexts is None'
        if hasattr(self, 'cond_proj'):
            attention_contexts = attention_contexts*torch.exp(self.cond_proj(attention_contexts))
        
        z_mu, z_logsigma = self.get_z(x, z_acts, attention_contexts)# -> [B, latent_dim], [B, latent_dim]
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)# -> [B, btl_dim, T], [B, latent_dim], [B, latent_dim]
        if use_pred_z:
            z_embed = self.get_z_embed(z_mu*0.01+zp_mu*0.99, z_logsigma*0.01+zp_logsigma*0.99)# -> [B, latent_dim]
        else:
            z_embed = self.get_z_embed(z_mu, z_logsigma)# -> [B, latent_dim]
        x = x + z_embed.unsqueeze(-1)# [B, btl_dim, T] -> [B, btl_dim, T]
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        x = self.resnet(x)
        if mel_mask is not None:
            x.masked_fill_(~mel_mask, 0.0)
        
        if self.use_z_variance:
            kl = gaussian_analytical_kl_var(z_mu, zp_mu, z_logsigma, zp_logsigma).unsqueeze(-1)# [B, latent_dim, T]
        else:
            kl = gaussian_analytical_kl(z_mu, zp_mu, z_logsigma, zp_logsigma).unsqueeze(-1)# -> [B, latent_dim, 1]
        return x, kl# [B, btl_dim, T], [B, latent_dim, 1]
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        assert attention_contexts is not None, 'attention_contexts is None'
        if hasattr(self, 'cond_proj'):
            attention_contexts = attention_contexts*torch.exp(self.cond_proj(attention_contexts))
        
        x, zp_mu, zp_logsigma = self.pred_z(x, attention_contexts)# -> [B, btl_dim, T], [B, latent_dim], [B, latent_dim]
        
        z_embed = self.get_z_embed(zp_mu, zp_logsigma)# -> [B, latent_dim]
        x = x + z_embed.unsqueeze(-1)# [B, btl_dim, T] -> [B, btl_dim, T]
        x.masked_fill_(~mel_mask, 0.0)
        
        x = self.resnet(x)
        x.masked_fill_(~mel_mask, 0.0)
        return x

class DecBlock(nnModule):# N*ResBlock + Downsample + N*TopDownBlock + Upsample + Conv1d Grouped Cond
    def __init__(self, hparams, mem_dim, hdn_dim, btl_dim, latent_dim, n_blocks, scale=1):# hparams, hidden_dim, bottleneck_dim, latent_dim
        super().__init__()
        self.mem_dim = mem_dim
        
        if mem_dim and hparams.exp_cond_proj:
            self.res_cond_proj = nn.Conv1d(mem_dim, btl_dim, 1)
            nn.init.zeros_(self.res_cond_proj.weight)
            nn.init.zeros_(self.res_cond_proj.bias)
            
            self.td_cond_proj = nn.Conv1d(mem_dim, btl_dim, 1)
            nn.init.zeros_(self.td_cond_proj.weight)
            nn.init.zeros_(self.td_cond_proj.bias)
        
        self.n_blocks = n_blocks
        self.tdblock = []
        self.melresnet = []
        for i in range(self.n_blocks):
            is_first_block = bool(i==0)
            self.melresnet.append(ResBlock(btl_dim+(mem_dim*is_first_block), hdn_dim, btl_dim, residual=True))
            self.tdblock.append(TopDownBlock(hparams, hdn_dim, btl_dim, latent_dim, mem_dim if is_first_block else 0))
        self.melresnet = nn.ModuleList(self.melresnet)
        self.tdblock   = nn.ModuleList(self.tdblock)
        
        self.scale = scale
    
    def downsample(self, x):
        return F.avg_pool1d(x, kernel_size=self.scale)
    
    def upsample(self, x):
        return F.interpolate(x, scale_factor=self.scale)
    
    def forward_up(self, z_acts, attention_contexts):
        if self.mem_dim and hasattr(self, 'res_cond_proj'):
            z_acts = z_acts*torch.exp(self.res_cond_proj(attention_contexts))
        
        z_acts = torch.cat((z_acts, attention_contexts), dim=1)
        for i in range(self.n_blocks):
            z_acts = self.melresnet[i](z_acts)
        return z_acts
    
    def forward(self, x, z_acts, attention_contexts, mel_mask, use_pred_z=False):#     x:'Comes from previous DecBlock'
                                                     #z_acts:'Comes from spect ResBlocks during training'
                                                     #attent:'Comes from the Text Encoder after being expanded'
        if self.mem_dim and hasattr(self, 'td_cond_proj'):
            x = x*torch.exp(self.td_cond_proj(attention_contexts))
        
        kls = []
        for i in range(self.n_blocks):
            x, kl = self.tdblock[i](x, z_acts, attention_contexts, mel_mask, use_pred_z=use_pred_z)
            kls.append(kl)
        return x, kls# [B, btl_dim, T], list([B, latent_dim, T])
    
    def infer(self, x, attention_contexts, mel_mask, std=None):
        if std is not None:
            self.std = std
        
        if self.mem_dim and hasattr(self, 'td_cond_proj'):
            x = x*torch.exp(self.td_cond_proj(attention_contexts))
        
        for i in range(self.n_blocks):
            x = self.tdblock[i].infer(x, attention_contexts, mel_mask, std)
        
        return x

class AutoregressiveLSTM(nnModule):
    def __init__(self, input_dim, hidden_dim, cond_dim=0, n_layers=1, dropout=0.0, zoneout=0.1):
        super(AutoregressiveLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.zoneout = zoneout
        
        inp_dim = input_dim+cond_dim
        self.lstm_cell = []
        for i in range(self.lstm_n_layers):
            self.lstm_cell.append(LSTMCellWithZoneout(inp_dim, self.hidden_dim, dropout=self.dropout, zoneout=self.zoneout))
            inp_dim = self.hidden_dim
        self.lstm_cell = nn.ModuleList(self.lstm_cell)
        self.lstm_proj = ConvNorm(inp_dim, input_dim)
    
    def forward(self, x, x_lengths, cond=None):# [B, T, input_dim], [B], [B, T, cond_dim]
        if self.cond_dim:
            assert cond is not None, 'self.cond_dim > 0 but cond is None'
        assert x.shape[2] == self.input_dim
        B, input_dim, T = x.shape
        
        cond = cond.unbind(2)          # [B,  cond_dim, T] -> [[B,  cond_dim],]*T
        x = F.pad(x, (1, -1)).unbind(2)# [B, input_dim, T] -> [[B, input_dim],]*T
        
        states       = [(x[0].new_zeros(B, self.hidden_dim), x[0].new_zeros(B, self.hidden_dim)) for i in range(self.n_layers)]
        final_states = [[x[0].new_zeros(B, self.hidden_dim),
                         x[0].new_zeros(B, self.hidden_dim)] for i in range(self.n_layers)]
        
        pred_x = []
        for i, (xi, ci) in enumerate(zip(x, cond)):
            final_idx = (x_lengths==i+1)
            final_idx_sum = final_idx.sum()
            
            xi = torch.cat((xi, ci), dim=1)# [B, input_dim], [B, cond_dim] -> [B, input_dim+cond_dim]
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(xi, states[j])
                xi = states[j][0]
                if final_idx_sum:
                    final_states[j][0] = torch.where(final_idx.unsqueeze(1), states[j][0], final_states[j][0])
                    final_states[j][1] = torch.where(final_idx.unsqueeze(1), states[j][1], final_states[j][1])
            pred_x.append(self.lstm_proj(xi))# .append([B, input_dim])
        pred_x = torch.stack(pred_x, dim=2)# [B, input_dim, T]
        pred_x = pred_x*get_mask_from_lengths(x_lengths).unsqueeze(1)
        
        return pred_x
    
    def infer(self, x_lengths, cond=None):# [B], [B, T, cond_dim]
        states = [(x[0].new_zeros(B, self.hidden_dim), x[0].new_zeros(B, self.hidden_dim)) for i in range(self.n_layers)]
        xi = x[0].new_zeros(B, self.input_dim)
        
        pred_x = []
        for i, ci in enumerate(cond):
            xi = torch.cat((xi, ci), dim=1)# [B, input_dim], [B, cond_dim] -> [B, input_dim+cond_dim]
            for j, lstm_cell in enumerate(self.lstm_cell):
                states[j] = lstm_cell(xi, states[j])
                xi = states[j][0]
            xi = self.lstm_proj(xi)
            pred_x.append(xi)# .append([B, input_dim])
        pred_x = torch.stack(pred_x, dim=2)# [B, input_dim, T]
        pred_x = pred_x*get_mask_from_lengths(x_lengths).unsqueeze(1)
        
        return pred_x

class Decoder(nnModule):
    def __init__(self, hparams, mem_dim, hdn_dim, btl_dim, latent_dim):# hparams, memory_dim, hidden_dim, bottleneck_dim, latent_dim
        super(Decoder, self).__init__()
        self.memory_efficient = False#hparams.memory_efficient
        self.use_z_variance = getattr(hparams, 'use_z_variance', False)
        self.use_pred_z = False
        self.n_blocks = hparams.decoder_n_blocks
        self.scale = 2
        self.downscales = [self.scale**i for i in range(self.n_blocks)]
        
        self.start = nn.Conv1d(hparams.n_mel_channels, btl_dim, 1)
        
        # project x -> spect
        self.spkr_proj = []
        for i in range(self.n_blocks):
            self.spkr_proj.append(LinearNorm(hparams.speaker_embedding_dim, mem_dim, bias=True))
        self.spkr_proj = nn.ModuleList(self.spkr_proj)
        
        # Decoder Blocks (one per timescale)
        # each contains ResBlock, TopDownBlock, upsample func, downsample func
        self.block = []
        for i in range(self.n_blocks):
            n_blocks = hparams.n_blocks_per_timescale if type(hparams.n_blocks_per_timescale) is int else hparams.n_blocks_per_timescale[i]
            self.block.append(DecBlock(hparams, mem_dim, hdn_dim, btl_dim, latent_dim, n_blocks, scale=self.scale))
        self.block = nn.ModuleList(self.block)
        
        self.lstm_up   = LSTMBlock(mem_dim+btl_dim, hdn_dim, btl_dim, out_vec_width=None, residual=False, output_vector=False)
        self.lstm_down = TopDownLSTMBlock(hparams, 0, hdn_dim, btl_dim, latent_dim, mem_dim)
        
        # project x -> spect
        self.proj = []
        for i in range(self.n_blocks):
            self.proj.append(nn.Conv1d(btl_dim, hparams.n_mel_channels, 1, bias=True))
        self.proj = nn.ModuleList(self.proj)
    
    def maybe_cp(self, func, *args):
        func_callable = func.__call__ if inspect.isclass(func) else func
        if self.memory_efficient and self.training:
            return checkpoint(func_callable, *args)
        else:
            return func_callable(*args)
    
    def downsample_to_list(self, x):
        x_list = []
        for i, block in enumerate(self.block):
            x_list.append(x)
            
            is_last_block = (i+1==len(self.block))
            if not is_last_block:
                x = block.downsample(x)
        return x_list
    
    def get_mask_list(self, mel_lengths, type='floor'):
        mel_mask = get_mask_from_lengths(mel_lengths).unsqueeze(1)# [B, 1, mel_T]
        mel_mask = F.pad(mel_mask, (0, self.downscales[-1]-mel_mask.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        
        mel_masks = self.downsample_to_list(mel_mask.float())
        if   type=='floor':
            mel_masks = [mask.floor().bool() for mask in mel_masks]
        elif type=='round':
            mel_masks = [mask.round().bool() for mask in mel_masks]
        elif type== 'ceil':
            mel_masks = [mask. ceil().bool() for mask in mel_masks]
        else:
            raise NotImplementedError
        return mel_mask, mel_masks
    
    def forward_up(self, gt_mel, *attention_contexts_list):
        gt_mel = F.pad(gt_mel, (0, self.downscales[-1]-gt_mel.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        z_acts = self.start(gt_mel)# [B, btl_dim, mel_T]
        z_acts_list = []
        for i, block in enumerate(self.block):# go up the ResBlock's
            z_acts = block.forward_up(z_acts, attention_contexts_list[i])
            if torch.isinf(z_acts).any() or torch.isnan(z_acts).any():
                print(f"Up ResBlock {i} has a nan or inf output.")
            z_acts_list.append(z_acts)
            
            is_last_block = (i+1==len(self.block))
            if not is_last_block:
                z_acts = block.downsample(z_acts)
        
        z_acts = torch.cat((attention_contexts_list[-1], z_acts), dim=1)
        z_acts_top = self.lstm_up(z_acts)# -> [B, btl_dim, mel_T]
        if torch.isinf(z_acts).any() or torch.isnan(z_acts).any():
            print(f"Up LSTMBlock has a nan or inf output.")
        return (z_acts_top, *z_acts_list)
    
    def forward_down(self, z_acts_top, z_acts_list, attention_contexts_list, mel_masks, use_pred_z=False):
        use_pred_z = use_pred_z or self.use_pred_z
        x, kl = self.lstm_down(None, z_acts_top, attention_contexts_list[-1], mel_masks[-1])
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(f"TopDownLSTMBlock has a nan or inf output.")
        
        x_list  = [ x,]
        kl_list = [kl,]
        for i, block in reversed(list(enumerate(self.block))):# go down the TopDownBlock stack
            x, kls = block(x, z_acts_list[i], attention_contexts_list[i], mel_masks[i], use_pred_z=use_pred_z)
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"TopDownBlock {i} has a nan or inf output.")
            x_list.append(x)
            kl_list.extend(kls)
            
            is_first_block = (  i==0  )
            is_last_block  = (i+1==len(self.block))
            if not is_first_block:
                x = block.upsample(x)
        x_list  =  x_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        kl_list = kl_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        
        mel_list = []
        for i, proj in enumerate(self.proj):
            mel_list.append(proj(x_list[i]))
        # mel_list -> [bottom, ..., top,]
        
        return (*x_list, *mel_list, *kl_list)
    
    def forward(self, gt_mel, attention_contexts, mel_lengths, speaker_embed, use_pred_z=False):
        B,     n_mel, mel_T = gt_mel.shape
        B,   mem_dim, mel_T = attention_contexts.shape
        B, embed_dim        = speaker_embed.shape
        
        mel_mask, mel_masks = self.get_mask_list(mel_lengths, type='ceil')# [B, 1, mel_T], list([B, 1, mel_T//2**i] for i in range(self.n_blocks))
        
        # downsample attention_contexts
        attention_contexts = F.pad(attention_contexts, (0, self.downscales[-1]-attention_contexts.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        attention_contexts_list = self.downsample_to_list(attention_contexts)
        
        # add seperate speaker info for each attention_contexts timescale
        attention_contexts_list = [x+self.spkr_proj[i](speaker_embed).unsqueeze(2) for i, x in enumerate(attention_contexts_list)]
        
        z_acts_top, *z_acts_list = self.maybe_cp(self.forward_up, *(gt_mel, *attention_contexts_list))
        
        out = self.maybe_cp(self.forward_down, *(z_acts_top, z_acts_list, attention_contexts_list, mel_masks, use_pred_z))
        x_list, mel_list, kl_list = self.group_returned_tuple(out)
        
        return (*x_list, *mel_list, *kl_list)
    
    def group_returned_tuple(self, x):
        x_list   = x[                0:1*self.n_blocks+1]
        mel_list = x[1*self.n_blocks+1:2*self.n_blocks+1]
        kl_list  = x[2*self.n_blocks+1:                 ]
        return x_list, mel_list, kl_list
    
    def infer(self, attention_contexts, mel_lengths, speaker_embed, std, global_std=None):
        B,   mem_dim, mel_T = attention_contexts.shape
        B, embed_dim        = speaker_embed.shape
        
        mel_mask, mel_masks = self.get_mask_list(mel_lengths, type='ceil')# [B, 1, mel_T], list([B, 1, mel_T//2**i] for i in range(self.n_blocks))
        
        # downsample attention_contexts
        attention_contexts = F.pad(attention_contexts, (0, self.downscales[-1]-attention_contexts.shape[-1]%self.downscales[-1]))# pad to multiple of max downscale
        attention_contexts_list = self.downsample_to_list(attention_contexts)
        
        # add seperate speaker info for each attention_contexts timescale
        attention_contexts_list = [x+self.spkr_proj[i](speaker_embed).unsqueeze(2) for i, x in enumerate(attention_contexts_list)]
        
        x = self.lstm_down.infer(None, attention_contexts_list[-1], mel_masks[-1], std)
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(f"TopDownLSTMBlock has a nan or inf output.")
        
        x_list  = [x,]
        for i, block in reversed(list(enumerate(self.block))):# go down the TopDownBlock stack, from last (highest) to first (lowest)
            is_first_block = (  i==0  )
            is_last_block  = (i+1==len(self.block))
            
            x = block.infer(x, attention_contexts_list[i], mel_masks[i],
                                global_std if (i>=len(self.block)-2 and global_std is not None) else std)# use global_std for top 2 blocks
            if torch.isinf(x).any() or torch.isnan(x).any():
                print(f"TopDownBlock {i} has a nan or inf output.")
            x_list.append(x)
            
            if not is_first_block:
                x = block.upsample(x)
        x_list = x_list[::-1]# reverse list -> [bottom, ..., top, toplstm]
        
        mel_list = []
        for i, proj in enumerate(self.proj):
            mel_list.append(proj(x_list[i]))
        # mel_list -> [bottom, ..., top,]
        
        pred_mel = mel_list[0]
        return pred_mel, x_list[0]# [B, n_mel, mel_T]
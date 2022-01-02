import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@torch.jit.script
def viterbi(log_probs: Tensor, enc_lens: Tensor, dec_lens: Tensor, n_rows_forward: int = 1, pad_mag: float = 1e12) -> Tensor:
    """takes log probability Tensor of shape [B, enc_T, dec_T] and returns highest probability path as alignment tensor (same shape)."""
    B, enc_T, dec_T = log_probs.size()
    log_beta = torch.ones(B, enc_T, dec_T, device=log_probs.device, dtype=log_probs.dtype) * (-pad_mag)
    log_beta[:, 0, 0] = log_probs[:, 0, 0]
    
    for t in range(1, dec_T):
        step_log_beta = log_beta[:, :, t-1:t]
        prev_step = torch.cat(
            [F.pad(step_log_beta, (0, 0, n, -n), value=-pad_mag) for n in range(n_rows_forward+1)], dim=-1,
        ).max(dim=-1)[0]
        log_beta[:, :, t] = prev_step + log_probs[:, :, t]
    
    B_idx = torch.arange(B)
    curr_rows = enc_lens -1
    curr_cols = dec_lens -1
    path = [curr_rows * 1.0]
    for _ in range(dec_T -1):
        next_cols = (curr_cols -1).to(torch.long)
        go_dist = torch.stack([log_beta[B_idx, (curr_rows -n).to(torch.long), next_cols] for n in range(n_rows_forward+1)], dim=-1).argmax(dim=-1)
        curr_rows = (curr_rows -go_dist.to(curr_rows)).clamp_(min=0)
        curr_cols.sub_(1).clamp_(min=0)
        path.append(curr_rows)
    
    path.reverse()
    path = torch.stack(path, -1)
    
    indices = torch.arange(path.max() + 1).view(1, 1, -1).to(path)  # 1, 1, dec_T
    align = (indices == path.unsqueeze(-1)).to(path)  # B, enc_T, dec_T
    
    for i in range(B):
        pad = dec_T - int(dec_lens[i].item())
        align[i] = F.pad(align[i], (0, 0, -pad, pad))
    
    return align.transpose(1, 2)  # [B, enc_T, dec_T]

if __name__ == '__main__':
    from typing import Optional, List, Tuple
    import matplotlib
    from matplotlib.pyplot import yticks
    import matplotlib.pylab as plt
    import numpy as np
    import time
    
    def set_range(im, clim: Tuple[float, float]):
        if clim is not None:
            assert len(clim) == 2, 'range params should be a 2 element List of [Min, Max].'
            assert clim[1] > clim[0], 'Max (element 1) must be greater than Min (element 0).'
            im.set_clim(clim[0], clim[1])
    
    def plot_alignment_to_numpy(
            alignment: np.ndarray, # [txt_T, mel_T]
            text_symbols = None,
            info: str = None,
            title: str = None,
            clim = None):
        if clim is None:
            clim = [0.0, 1.0]
        fig, ax = plt.subplots(figsize=(12, 6), )
        im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                       interpolation='none')
        set_range(im, clim)
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        if info is not None:
            xlabel += '\n' + info
        plt.xlabel(xlabel)
        plt.ylabel('Encoder timestep')
        if title is not None:
            plt.title(title)
        if text_symbols is not None:
            yticks(range(alignment.shape[0]), text_symbols)
        plt.tight_layout()
        plt.draw()
    
    @torch.jit.script
    def viterbi_new(log_probs: Tensor, enc_lens: Tensor, dec_lens: Tensor, n_rows_forward: int = 1, pad_mag: float = 1e12) -> Tensor:
        """takes log probability Tensor of shape [B, enc_T, dec_T] and returns highest probability path as alignment tensor (same shape)."""
        B, enc_T, dec_T = log_probs.size()
        log_beta = torch.ones(B, enc_T, dec_T, device=log_probs.device, dtype=log_probs.dtype) * (-pad_mag)
        log_beta[:, 0, 0] = log_probs[:, 0, 0]
        
        #start_time = time.time()
        for t in range(1, dec_T):
            step_log_beta = log_beta[:, :, t-1:t]
            prev_step = torch.cat(
                [F.pad(step_log_beta, (0, 0, n, -n), value=-pad_mag) for n in range(n_rows_forward+1)], dim=-1,
            ).max(dim=-1)[0]
            log_beta[:, :, t] = prev_step + log_probs[:, :, t]
        #print('96: ', time.time() - start_time)
        
        #start_time = time.time()
        B_idx = torch.arange(B)
        curr_rows = enc_lens -1 # [B]
        curr_cols = dec_lens -1 # [B]
        path = [curr_rows * 1.0]
        for _ in range(dec_T -1):# iterate backwards through log_beta
            next_cols = (curr_cols -1).to(torch.long) # [B] constant change
            
            # log_beta.shape: [B, enc_T, dec_T] 
            go_dist = [log_beta[B_idx, (curr_rows -n).to(torch.long), next_cols] for n in range(n_rows_forward+1)]
            go_dist = torch.stack(go_dist, dim=-1).argmax(dim=-1)
            
            curr_rows = (curr_rows -go_dist.to(curr_rows)).clamp_(min=0) # [B] variable change
            curr_cols.sub_(1).clamp_(min=0) # [B] constant change
            path.append(curr_rows)
        path.reverse()
        #print('109: ', time.time() - start_time)
        
        path = torch.stack(path, -1)# [B, txt_T]*mel_T -> [B, txt_T, mel_T]
        indices = torch.arange(path.max() + 1).view(1, 1, -1).to(path)  # 1, 1, dec_T
        align = (indices == path.unsqueeze(-1)).to(path)  # B, enc_T, dec_T
        
        for i in range(B):
            pad = dec_T - int(dec_lens[i].item())
            align[i] = F.pad(align[i], (0, 0, -pad, pad))
        
        return align.transpose(1, 2)  # [B, enc_T, dec_T]
    
    enc_lens = torch.tensor([250])
    dec_lens = torch.tensor([800])
    alignment = torch.rand(enc_lens.max(), dec_lens.max())# [txt_T, mel_T]
    alignment[4, 100:300] = 1.0
    alignment = F.softmax(alignment*10.0, dim=0)# [txt_T, mel_T]
    #alignment[:, :5] = 0.0
    plot_alignment_to_numpy(alignment)
    
    hard_alignment_target = viterbi(alignment.log().unsqueeze(0), enc_lens, dec_lens, n_rows_forward=2, pad_mag=1e12).squeeze(0)
    hard_alignment = viterbi_new(alignment.log().unsqueeze(0), enc_lens, dec_lens, n_rows_forward=2, pad_mag=1e12).squeeze(0)
    assert (hard_alignment_target == hard_alignment).all()
    print("OUTPUTS MATCH")
    
    # OLD TIME 2nr:  93ms per item
    # OLD TIME 1nr:  78ms per item
    # OLD TIME 1nr: 178ms per 16 item batch
    
    start_time = time.time()
    for _ in range(64):
        hard_alignment = viterbi(alignment.log().unsqueeze(0).repeat(16, 1, 1), enc_lens.repeat(16), dec_lens.repeat(16), n_rows_forward=1, pad_mag=1e12).squeeze(0)
    print(f'Time per Item: {(time.time() - start_time)*(1000.0/64.0):.4f}ms')
    plot_alignment_to_numpy(hard_alignment)
import torch
import torch.nn.functional as F
from TSHP.utils.modules.utils import get_mask1d


def alignment_metric(alignments, input_lengths=None, output_lengths=None, enc_min_thresh=0.7, average_across_batch=False, adjacent_topk=True):
    """
    Diagonality = Network distance / Euclidean Distance
    https://media.springernature.com/lw685/springer-static/image/art%3A10.1186%2F1476-072X-7-7/MediaObjects/12942_2007_Article_201_Fig2_HTML.jpg
    
    Avg Max Attention = Average of Maximum Attention tokens at each timestep - Roughly equivalent to confidence between text and audio in TTS.
    Avg Top2 Attention = Average of Top 2 Attention tokens summed at each timestep. Can be better than Max because some symbols like "," and " " blend together. The model will not be better by learning the difference between "," and " " in the docoder so there's no reason to incentivize it.
    Avg Top3 Attention = Average of Top 3 Attention tokens summed at each timestep. Not tested, I'm guessing this also correlates with stability but I don't know how well.
    
    Encoder Max duration = Maximum timesteps spent on a single encoder token. If too much time is spent on a single token then that normally means the TTS model has gotten stuck on a single phoneme.
    Encoder Min duration = Minimum timesteps spent on a single encoder token. Can correlate with missing some letters or mis-pronouncing a word. The correlation is weak however so not really recommended for most models.
    Encoder Avg duration = Average timesteps spent on all (non-padded) encoder tokens. This value is equivalent to the speaking rate.
    
    p_missing_enc = Fraction of encoder tokens that had less summed alignment than enc_min_thresh. Used to identify if parts of the text were skipped.
    """
    alignments = alignments.clone()# [B, enc, dec]
    # alignments [batch size, x, y]
    # input_lengths [batch size] for len_x
    # output_lengths [batch size] for len_y
    if input_lengths is None:
        input_lengths  = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[1]-1) # [B] # 147
    if output_lengths is None:
        output_lengths = torch.ones(alignments.size(0), device=alignments.device)*(alignments.shape[2]-1) # [B] # 767
    batch_size = alignments.size(0)
    euclidean_distance = torch.sqrt(input_lengths.double().pow(2) + output_lengths.double().pow(2)).view(batch_size)
    
    # [B, enc, dec] -> [B, dec], [B, dec]
    max_values, cur_idxs = torch.max(alignments, 1) # get max value in column and location of max value
    
    cur_idxs = cur_idxs.float()
    prev_indx = torch.cat((cur_idxs[:, 0][:, None], cur_idxs[:, :-1]), dim=1)# shift entire tensor by one.
    dist = ((prev_indx - cur_idxs).pow(2) + 1).pow(0.5) # [B, dec]
    dist.masked_fill_(~get_mask1d(output_lengths, max_len=dist.shape[1]).squeeze(2), 0.0)# set dist of padded to zero
    dist = dist.sum(dim=1) # get total Network distance for each alignment
    diagonalitys = (dist + 1.4142135)/euclidean_distance # Network distance / Euclidean dist
    
    alignments.masked_fill_(~get_mask1d(output_lengths, max_len=alignments.shape[2]).squeeze(2)[:, None, :], 0.0)
    attm_enc_total = torch.sum(alignments, dim=2)# [B, enc, dec] -> [B, enc]
    
    # calc max  encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask1d(input_lengths, max_len=attm_enc_total.shape[1]).squeeze(2), 0.0)
    encoder_max_dur = attm_enc_total.max(dim=1)[0] # [B, enc] -> [B]
    
    # calc mean encoder durations (with padding ignored)
    encoder_avg_dur = attm_enc_total.mean(dim=1)   # [B, enc] -> [B]
    encoder_avg_dur *= (attm_enc_total.shape[1]/input_lengths.float())
    
    # calc min encoder durations (with padding ignored)
    attm_enc_total.masked_fill_(~get_mask1d(input_lengths, max_len=attm_enc_total.shape[1]).squeeze(2), 1.0)
    encoder_min_dur = attm_enc_total.min(dim=1)[0] # [B, enc] -> [B]
    
    # calc average max attention (with padding ignored)
    max_values.masked_fill_(~get_mask1d(output_lengths, max_len=max_values.shape[1]).squeeze(2), 0.0) # because padding
    avg_prob = max_values.mean(dim=1)
    avg_prob *= (alignments.shape[2]/output_lengths.float()) # because padding

    top2_avg_prob = get_top2_avg_prob(adjacent_topk, alignments, output_lengths)# todo, convert into general func
    top3_avg_prob = get_top3_avg_prob(adjacent_topk, alignments, output_lengths)# todo, convert into general func
    
    # calc portion of encoder durations under min threshold
    attm_enc_total.masked_fill_(~get_mask1d(input_lengths, max_len=attm_enc_total.shape[1]).squeeze(2), float(1e3))
    p_missing_enc = (torch.sum(attm_enc_total < enc_min_thresh, dim=1)) / input_lengths.float()
    
    diagonality_err = (diagonalitys-1.07).abs()# ideal value of 1.07 found empirically
    
    out = {
        "diagonality": diagonalitys,
        "diagonality_err": diagonality_err,
        "top1_avg_prob": avg_prob,
        "top2_avg_prob": top2_avg_prob,
        "top3_avg_prob": top3_avg_prob,
        "encoder_max_dur": encoder_max_dur,
        "encoder_min_dur": encoder_min_dur,
        "encoder_avg_dur": encoder_avg_dur,
        "p_missing_enc": p_missing_enc
    }
    if average_across_batch:
        out = {k: v.mean() for k, v in out.items()}
    return out


def get_top2_avg_prob(adjacent_topk, alignments, output_lengths):
    # calc average top2 attention (with padding ignored)
    if adjacent_topk:
        alignment_summed = alignments + F.pad(alignments, (0, 0, 1, -1,))  # [B, enc, dec]
        top_vals = torch.max(alignment_summed, dim=1)[0]  # -> [B, dec]
    else:
        top_vals = torch.topk(alignments, k=2, dim=1, largest=True, sorted=True)[0].sum(dim=1)  # [B, enc, dec] -> [B, dec]
    top2_avg_prob = top_vals.mean(dim=1)
    top2_avg_prob *= (alignments.shape[2] / output_lengths.float())  # because padding
    return top2_avg_prob


def get_top3_avg_prob(adjacent_topk, alignments, output_lengths):
    # calc average top3 attention (with padding ignored)
    if adjacent_topk:
        alignment_summed = alignments + F.pad(alignments, (0, 0, 1, -1,)) + F.pad(alignments, (0, 0, -1, 1,))  # [B, enc, dec]
        top_vals = torch.max(alignment_summed, dim=1)[0]  # -> [B, dec]
    else:
        top_vals = torch.topk(alignments, k=3, dim=1, largest=True, sorted=True)[0].sum(dim=1)  # [B, enc, dec] -> [B, dec]
    top3_avg_prob = top_vals.mean(dim=1)
    top3_avg_prob *= (alignments.shape[2] / output_lengths.float())  # because padding
    return top3_avg_prob

# imports
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from TSHP.utils.modules.loss_func.common import get_gate_BCE, get_mean_errors, get_CTC_loss
from TSHP.utils.modules.loss_func.guided_attention import GuidedAttentionLoss

from TSHP.utils.misc_utils import zip_equal

from TSHP.utils.modules.decoders.rnn_att_scores import alignment_metric
from TSHP.utils.modules.decoders.tacotron2 import RecurrentBlock

from TSHP.utils.arg_utils import force, get_args, force_any
from TSHP.utils.modules.utils import get_mask1d, dropout_frame, get_first_over_thresh, Fpad

from TSHP.utils.modules.core import nnModule, ModelModule, DictAsMember, ResBlock
from TSHP.utils.modules.embeddings import SpeakerEmbedding
from TSHP.utils.modules.text_encoder import TextModule


class Generator(nnModule):
    def __init__(self, h, mh):
        super().__init__()
        self.n_mel   = h['dataloader_config']['stft_config']['n_mel']
        self.hop_len = h['dataloader_config']['stft_config']['hop_len']
        self.pos_weight = 10.0
        
        self.spkrmodule = SpeakerEmbedding(mh)
        self.textmodule = TextModule(
            DictAsMember(mh.textmodule), mh.n_symbols,
            mh.text_embed_dim, mh.decoder.att_value_dim,
            moji_dim=2304,
            bert_dim=h['dataloader_config']['bert_config']['bert_embed_dim'],
            cond_dim=mh.speaker_embed_dim,
        )
        
        self.dfr = mh.drop_frame_rate
        self.tfr = mh.teacher_force_rate
        self.n_frames_per_step = mh.decoder.n_frames_per_step
        self.decoder = RecurrentBlock(
            DictAsMember(mh.decoder), self.n_mel,
            cond_dim=0, randn_dim=0,
            proj_logvar=False, output_dim=self.n_mel+1)
        
        # project hard alignment context to mel-spectrogram
        self.att_meldec = ResBlock(
            mh.decoder.att_value_dim, self.n_mel//4, self.n_mel,
            n_blocks=2, n_layers=2,
            kernel_size=1, act_func='relu',
        )
        
        self.use_postnet = mh.use_postnet
        if self.use_postnet:
            self.postnet = ResBlock(self.n_mel, self.n_mel, cond_dim=mh.speaker_embed_dim, **mh.postnet_config)
    
    def split_melgame(self, melgate):
        pr_mel  = melgate[..., :-1]# [B, ..., n_mel]
        pr_gate = melgate[..., -1:]# [B, ...,     1]
        return pr_mel, pr_gate
    
    def reshape_outputs(self, pr_melgate, alignments):
        return (*self.split_melgame(pr_melgate), torch.cat(alignments, dim=2)) # ([B, mel_T, n_mel], [B, mel_T, 1], [B, txt_T, mel_T])
    
    def loss(self, gt_mel, pr_mel, pr_mel_p, pr_gate_logits,
            gt_att_mel, pr_att_mel,
            alignments, mel_lens, txt_lens, align_mel_lens=None):
        loss_dict = {}
        loss_dict[ 'mel_MAE'], loss_dict[ 'mel_MSE'] = get_mean_errors(gt_mel, pr_mel  , mel_lens)
        if pr_mel_p is not None:
            loss_dict['melp_MAE'], loss_dict['melp_MSE'] = get_mean_errors(gt_mel, pr_mel_p, mel_lens)
        loss_dict['attmel_MAE'], loss_dict['attmel_MSE'] = get_mean_errors(gt_att_mel, pr_att_mel, mel_lens)
        
        loss_dict['gate_BCE'] = get_gate_BCE(pr_gate_logits, mel_lens, self.pos_weight)
        
        if align_mel_lens is not None:
            stp_lens = (align_mel_lens/self.n_frames_per_step).floor().long()
        else:
            stp_lens = (      mel_lens/self.n_frames_per_step).floor().long()
        
        # alignments.shape = [B, txt_T, mel_T]
        # add {'diagonalitys', 'top1_avg_prob', 'top2_avg_prob', 'top3_avg_prob', 'encoder_max_dur', 'encoder_min_dur', 'encoder_avg_dur', 'p_missing_enc'}
        loss_dict.update(
            alignment_metric(
                alignments, input_lengths=txt_lens.view(-1), output_lengths=stp_lens.view(-1), enc_min_thresh=0.7/self.n_frames_per_step,
            )
        )
        # add {'diag_att_loss',}
        loss_dict['diag_att_MAE'], loss_dict['diag_att_MSE'] = GuidedAttentionLoss(sigma=0.6).forward(alignments.transpose(1, 2), txt_lens, stp_lens)
        
        loss_dict['att_start_MAE'], loss_dict['att_start_MSE'] = get_mean_errors(alignments[:,          0,          0].view(-1, 1, 1), torch.tensor(1.0, device=alignments.device).to(alignments).view(1, 1, 1))
        loss_dict['att_end_MAE'  ], loss_dict['att_end_MSE'  ] = get_mean_errors(alignments[:, txt_lens-1, mel_lens-1].view(-1, 1, 1), torch.tensor(1.0, device=alignments.device).to(alignments).view(1, 1, 1))
        
        loss_dict['att_min_MAE'], loss_dict['att_min_MSE'] = get_mean_errors(
            torch.min(alignments.sum(2, True), torch.tensor(1.0).to(alignments)),
            torch.tensor(1.0).to(alignments), txt_lens, rescale_grad=False)
        
        #alignments = alignments[:, 1:, :] # [B, txt_T, mel_T] # strip start token (should be updated to use start token length from config file)
        text_order = torch.arange(alignments.shape[1], device=gt_mel.device, dtype=gt_mel.dtype)[None, :].expand(gt_mel.shape[0], alignments.shape[1])
        text_order = (text_order+1).masked_fill_(~get_mask1d(txt_lens, squ=True), 0)
        loss_dict['att_CTC'] = get_CTC_loss(
            Fpad(alignments.float(), (1, 0)).transpose(1, 2).add_(1e-6).log(),
                text_order.long(), stp_lens, txt_lens, blank_idx=0, zero_infinity=True)
        
        return loss_dict
    
    def get_blank_frame(self, batch_size):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.n_mel*self.n_frames_per_step, device=device, dtype=dtype)
    
    def init_attention(self, spkr_ids, text_ids, moji_embed, bert_embed, txt_lens):
        spkr_embed = self.spkrmodule(spkr_ids)
        text_acts, kld = self.textmodule(text_ids, moji_embed, bert_embed, txt_lens, cond=spkr_embed)
        self.decoder.update_kv(text_acts, txt_lens, spkr_embed)
        return text_acts, spkr_embed, kld
    
    def infer(
            self,
            spkr_ids,
            text_ids, moji_embed, bert_embed, txt_lens,
            max_decode_length=9999, gate_threshold=0.5,
        ):
        spkr_ids, text_ids, moji_embed, bert_embed, txt_lens = self.transfer_device([spkr_ids, text_ids, moji_embed, bert_embed, txt_lens])
        text_acts, spkr_embed, kld = self.init_attention(spkr_ids, text_ids, moji_embed, bert_embed, txt_lens)
        B, txt_T, *_ = text_ids.shape
        
        gt_mel_frame = self.get_blank_frame(B)
        
        states = None; attention_output = None; max_gate = torch.zeros(B)
        pred_melgates = []; alignments = []; i = -1
        for i in range(max_decode_length):
            pred_melgate, attention_output, alignment, states = self.decoder.ar_forward(gt_mel_frame, attention_output, states, spkr_embed)# [B, H], TensorList(), [B, txt_T, C] -> [B, n_mel+1], [B, mel_T, txt_T], TensorList()
            pred_melgates.append(pred_melgate)# list.append([B, n_fps, output_dim])
            alignments   .append(alignment)   # list.append([B, txt_T, 1])
            
            pred_melgate = pred_melgate.view(pred_melgate.shape[0], self.n_frames_per_step, self.n_mel+1)# -> [B, n_frames_per_step, n_mel+1]
            
            gt_mel_frame = pred_melgate[:, :, :-1].reshape(pred_melgate.shape[0], self.n_frames_per_step*self.n_mel)# -> [B, n_frames_per_step*n_mel]
            max_gate = torch.max(torch.max(pred_melgate[:, :, -1].data.cpu().float(), dim=1)[0].sigmoid(), max_gate)# -> [B]
            if max_gate.min() > gate_threshold and i > 2:
                break
        else:# if loop finishes without breaking
            print(f"WARNING: Decoded {i+1} steps without finishing speaking. Breaking loop.")
        self.decoder.reset_kv()
        pred_melgate = torch.cat(pred_melgates, dim=1)# -> [B, mel_T, (n_mel+1)]
        pr_mel, pr_gate, alignments = self.reshape_outputs(pred_melgate, alignments)
        
        mel_lens = get_first_over_thresh(pr_gate, gate_threshold).view(-1, 1, 1)
        mel_T = mel_lens.max().item()
        pr_mel = pr_mel[:, :mel_T]
        alignments = alignments[:, :, :mel_T]
        
        if self.use_postnet:
            pr_mel = self.postnet(pr_mel, cond=spkr_embed, lens=mel_lens)
        
        return {
            'pr_mel': pr_mel,# [B, mel_T, n_mel]
            'alignments': alignments, # [B, txt_T, mel_T]
            'mel_lens': mel_lens, # [B, 1, 1]
        }
    
    def forward(
            self, text,
            text_ids, moji_embed, bert_embed, txt_lens,
            gt_mel, mel_lens, spkr_ids,
            inference_mode=False):
        text_acts, spkr_embed, kld = self.init_attention(spkr_ids, text_ids, moji_embed, bert_embed, txt_lens)
        
        mel_T = gt_mel.shape[1] + (-gt_mel.shape[1])%self.n_frames_per_step# mel_T rounded up to nearest n_frames_per_step factor
        mel_mask = get_mask1d(mel_lens, max_len=mel_T)
        gt_mel_shifted = F.pad(gt_mel.transpose(1, 2), (self.n_frames_per_step, -self.n_frames_per_step+((-gt_mel.shape[1])%self.n_frames_per_step))).transpose(1, 2)
        if self.training:
            gt_mel_shifted = dropout_frame(gt_mel_shifted, mel_lens, self.dfr, soft_mask=True, smooth_range=9, max_len=mel_T)
        mel_feats = self.decoder.pre(gt_mel_shifted, spkr_embed, mel_mask)# -> [B, mel_T//n_frames_per_step, prenet_dim]
        
        tf_frames = (torch.rand(mel_feats.shape[1]) < self.tfr).tolist() # list[mel_T]
        
        tf_states = None; tf_attention_output = None
        declstm_outputs  = []; attention_outputs = []; alignments = []
        for i, (prenet_frame, b_tf) in enumerate(zip_equal(mel_feats.unbind(1), tf_frames)):
            tf_declstm_output, tf_attention_output, alignment, tf_states = self.decoder(prenet_frame, tf_attention_output, tf_states, spkr_embed)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            if (inference_mode or (not b_tf)) and i > 0:# inference
                self.decoder.prenet.eval()
                pr_melgate = self.decoder.post(declstm_output.detach(), attention_output.detach(), spkr_embed, None, reset_kv=False)# -> [B, n_frames_per_step, n_mel+1]
                fr_prenet_frame = self.decoder.pre(pr_melgate.detach()[:, :, :-1], spkr_embed, None)# -> [B, 1, prenet_dim]
                self.decoder.prenet.train(self.training)
                
                declstm_output, attention_output, alignment, states = self.decoder(fr_prenet_frame, attention_output, states, spkr_embed)# [B, H], TensorList(), [B, txt_T, C] -> [B, H], [B, H], TensorList()
            else:# teacher_force
                declstm_output, attention_output, states = tf_declstm_output, tf_attention_output, tf_states
            declstm_outputs.append(declstm_output); attention_outputs.append(attention_output); alignments.append(alignment)# list.append([B, 1, txt_T])
        
        pr_melgate = self.decoder.post(declstm_outputs, attention_outputs, spkr_embed, mel_mask, reset_kv=True)# -> [B, mel_T//n_frames_per_step, (n_mel+1)*n_frames_per_step]
        pr_mel, pr_gate_logits, alignments = self.reshape_outputs(pr_melgate, alignments)
        pr_mel  = pr_mel[:, :gt_mel.shape[1]]
        pr_gate_logits = pr_gate_logits[:, :gt_mel.shape[1]]
        
        hard_alignments = F.gumbel_softmax(alignments.add(1e-2).log(), dim=1)
        hard_att_context = hard_alignments.transpose(1, 2) @ text_acts # [B, txt_T, mel_T].t @ [B, txt_T, C]
        pr_att_mel = self.att_meldec(hard_att_context)
        
        gt_att_mel = F.avg_pool1d(gt_mel.exp().sqrt(), 4).pow(2).log() # [B, mel_T, n_mel] -> [B, mel_T, n_mel//4]
        
        pr_mel_p = None
        if self.use_postnet:
            pr_mel_p = self.postnet(pr_mel, cond=spkr_embed, lens=mel_lens)
        
        loss_dict_g = self.loss(
            gt_mel, pr_mel, pr_mel_p, pr_gate_logits,
            gt_att_mel, pr_att_mel,
            alignments, mel_lens, txt_lens
        )
        return {
            'loss_dict_g': loss_dict_g,# not reduced, each value's shape is [B]
            'pr_mel'     : pr_mel,     # [B, mel_T, n_mel]
            'pr_att_mel' : pr_att_mel, # [B, mel_T, n_mel//4]
            'pr_gate'    : pr_gate_logits.sigmoid(),# [B, mel_T,     1]
            'alignments' : alignments, # [B, txt_T, mel_T]
        }

class Model(ModelModule):
    def __init__(self, h):
        super().__init__()
        self.set_h(h)
        self.generator     = Generator    (self.h, self.mh)
#       self.discriminator = Discriminator(self.h, self.mh)
        
        self.diagonality_weight = 0.0
        self.diagonality_err_weight = 0.0
        self.top1_avg_prob_weight = 0.0
        self.top2_avg_prob_weight = 0.0
        self.top3_avg_prob_weight = 0.0
        self.encoder_max_dur_weight = 0.0
        self.encoder_min_dur_weight = 0.0
        self.encoder_avg_dur_weight = 0.0
        self.p_missing_enc_weight = 0.0
    
    def align(self, loss_weights, batch):
        out = self.transfer_device(batch)# transfer inputs to correct device and dtype
        out.update(force(self.generator, valid_kwargs=get_args(self.generator.forward), **out))
        out = self.loss_stuff(out, loss_weights)
        return out
    
    @torch.no_grad()
    def infer(self, batch):
        batch['spkr_ids'] = self.spkrnames_to_ids(batch['spkrname'])
        out = self.transfer_device(batch)# transfer inputs to correct device and dtype
        out.update(force(self.generator.infer, **out))
        return out
    
    def eval_infer(self, loss_weights, batch):
        """method called by TrainModule to evaluate a models inference performance."""
        out = self.transfer_device(batch)# transfer inputs to correct device and dtype
        with torch.no_grad():
            max_decode_length = int((min(self.h['filelist_config']['max_dur'], 1792)*self.h['audio_config']['sr'])/self.h['stft_config']['hop_len'])
            pr_mel, pr_gate, alignments = self.generator.infer(
                out['spkr_ids'],
                out['text_ids'], out['moji_embed'], out['bert_embed'], out['txt_lens'],
                max_decode_length=max_decode_length//self.generator.n_frames_per_step, gate_threshold=0.5,)
            assert pr_mel.shape[2] == self.generator.n_mel
            assert alignments.shape[1] == out['txt_lens'].max()
            
            mask_minus1 = get_mask1d(out['mel_lens']-1, max_len=out['mel_lens'].max())
            out['gt_gate'] = torch.logical_xor(mask_minus1, get_mask1d(out['mel_lens']))# [B, mel_T, 1]
            
            pr_mel_lens = get_first_over_thresh(pr_gate, 0.5)[:, None].long().clamp(min=2)# [B, 1, 1]
            out['pr_mel_lens'] = pr_mel_lens
            out['pr_mel']     = pr_mel    [:, :pr_mel_lens.max()]  # [B, mel_T, n_mel]
            out['pr_gate']    = pr_gate   [:, :pr_mel_lens.max()]  # [B, mel_T,     1]
            out['alignments'] = alignments[:, :out['txt_lens'].max(), :pr_mel_lens.max()]# [B, txt_T, mel_T]
            
            min_mel_lens = torch.min(pr_mel_lens, out['mel_lens'])
            
            out['loss_dict_g'] = self.generator.loss(
                out['gt_mel'][:, :min_mel_lens.max()], out['pr_mel'][:, :min_mel_lens.max()], pr_gate[:, :min_mel_lens.max()],
                alignments, min_mel_lens, out['txt_lens'], align_mel_lens=pr_mel_lens
            )
        
        out = self.loss_stuff(out, loss_weights)
        return out
    
    def forward(self, loss_weights, batch):
        out = self.transfer_device(batch)# transfer inputs to correct device and dtype
        
        # run modules
        if True:
            out.update(force(self.generator    , valid_kwargs=get_args(self.generator    .forward), **out))
        if hasattr(self, 'discriminator'):
            out.update(force(self.discriminator, valid_kwargs=get_args(self.discriminator.forward), **out))
        
        out = self.loss_stuff(out, loss_weights)
        return out
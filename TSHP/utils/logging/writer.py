import os
import random
import time
from shutil import copyfile
from typing import Union, Optional, List
from torch import Tensor

import torch

from TSHP.utils.modules.utils import Fpad
from TSHP.utils.saving.utils import safe_write
from scipy.io.wavfile import write

from TSHP.utils.dataset.audio.stft import mag_to_log
from TSHP.utils.logging.plotting import plot_spectrogram_to_numpy, plot_time_series_to_numpy, \
    plot_alignment_to_numpy

from torch.utils.tensorboard import SummaryWriter

try:# https://github.com/pytorch/pytorch/issues/30966#issuecomment-582747929
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
    pass

def get_pr_and_gt_versions(pr, k):
    gt_k = pr_k = None
    if 'gt_' in k:
        gt_k = k
        if k.replace('gt_', 'pr_') in pr.keys():
            pr_k = k.replace('gt_', 'pr_')
    elif 'pr_' in k:
        pr_k = k
        if k.replace('pr_', 'gt_') in pr.keys():
            gt_k = k.replace('pr_', 'gt_')
    else:
        gt_k = k
    return gt_k, pr_k

def is_feat_in_dict(k, pr):
    return bool(k in pr or 'gt_'+k in pr or 'pr_'+k in pr)

def write_wav(audio: torch.FloatTensor, path: str, sampling_rate: Union[int, float]):
    # scale audio for int16 output
    audio = (audio.float() * 2**15).squeeze().numpy().astype('int16')
    tmppath = f'{path}{random.randint(0, 9999)}.tmp'
    write(tmppath, sampling_rate, audio)# write to tmp path before moving so corruption doesn't occur if the write stops part-way-through
    os.rename(tmppath, path)

def maybe_load_model(ref, path):
    if not (ref or path):
        return None
    # import the model's code
    name = "Model"
    modelmodule = getattr(
        __import__(f'TSHP.models.{ref}.model', fromlist=[name]), name)
    # load checkpoint
    model, *_ = modelmodule.load_model(path, train=False)
    return model

def pad_same_length(*tensors):
    tensors_out = []
    max_T = max(t.shape[1] for t in tensors if t is not None)
    for tensor in tensors:
        if tensor is not None:
            tensors_out.append(Fpad(tensor, (0, max_T-tensor.shape[1])))
        else:
            tensors_out.append(tensor)
    return tuple(tensors_out)

def get_matched_lens(tensor: Tensor, lens_list: List[Tensor]):
    for lens in lens_list:
        if lens is None:
            continue
        if lens.max().item() == tensor.shape[1]:
            return lens
    else:
        raise RuntimeWarning(f"found no lens that match input tensor. input tensor len is {tensor.shape[1]}")

class Logger(SummaryWriter):
    keys_to_plot: dict
    terminal_config: dict
    n_plots: int
    def __init__(self, h, logdir, abtest_path, **kwargs):
        self.log_dir = logdir
        super(Logger, self).__init__(log_dir=self.log_dir, **h['metrics']['summary_writer_config'])
        self.abtest_path = abtest_path
        self.abtest_mellist_path = os.path.join(abtest_path, 'mellist.txt')
        self.abtest_wavlist_path = os.path.join(abtest_path, 'wavlist.txt')
        
        # config paths
        self.config_paths = [
            h['default_config_path'   ],
            h['model_config_path'     ],
            h['model_config_path_live'],
        ]
        
        # register config variables
        self.sr = h['audio_config']['sr']
        self.log_min_val = mag_to_log(float(h['stft_config']['clamp_val']), 0.0, h['stft_config']['log_type'])
        self.log_max_val = mag_to_log(                                60.0, 0.0, h['stft_config']['log_type'])
        self.amp_level = h['optimizer_config']['amp_level']
        
        for k, v in kwargs.items():# save config to self
            setattr(self, k, v)
        
        # init variables
        self.loss_dict_path = {}# log loss for each file
        self.enable_milliepoch_plots = h['metrics']['enable_milliepoch_plots']
        
        self.text_classifier = None
        self.spkr_classifier = None
        self.prosody_decoder = None
        self.vocoder         = None
        self.has_not_written_gt_wav = True
        
        self.running_loss_dict_mean = {}
        self.running_loss_dict_start_iteration = {}
        
        self.vocoder_config         = h['metrics']['vocoder_config']
        self.text_classifier_config = h['metrics']['text_classifier_config']
        self.spkr_classifier_config = h['metrics']['spkr_classifier_config']
        self.prosody_decoder_config = h['metrics']['prosody_decoder_config']
    
    def cpu(self):
        if self.vocoder is not None:
            self.vocoder        .cpu()
        if self.text_classifier is not None:
            self.text_classifier.cpu()
        if self.spkr_classifier is not None:
            self.spkr_classifier.cpu()
        if self.prosody_decoder is not None:
            self.prosody_decoder.cpu()
    
    def cuda(self):
        if self.vocoder is not None:
            self.vocoder        .cuda()
        if self.text_classifier is not None:
            self.text_classifier.cuda()
        if self.spkr_classifier is not None:
            self.spkr_classifier.cuda()
        if self.prosody_decoder is not None:
            self.prosody_decoder.cuda()
    
    def to(self, *args, **kwargs):
        if self.vocoder is not None:
            self.vocoder        .to(*args, **kwargs)
        if self.text_classifier is not None:
            self.text_classifier.to(*args, **kwargs)
        if self.spkr_classifier is not None:
            self.spkr_classifier.to(*args, **kwargs)
        if self.prosody_decoder is not None:
            self.prosody_decoder.to(*args, **kwargs)
    
    def export_configs(self):
        os.makedirs(self.log_dir    , exist_ok=True)
        os.makedirs(self.abtest_path, exist_ok=True)
        for config_path in self.config_paths:
            copyfile(
                config_path,
                os.path.join(self.log_dir, os.path.split(config_path)[-1])
            )
            copyfile(
                config_path,
                os.path.join(self.abtest_path, os.path.split(config_path)[-1])
            )
    
    def load_all_models(self):
        self.vocoder         = maybe_load_model(**self.vocoder_config)
        self.text_classifier = maybe_load_model(**self.text_classifier_config)
        self.spkr_classifier = maybe_load_model(**self.spkr_classifier_config)
        self.prosody_decoder = maybe_load_model(**self.prosody_decoder_config)
    
    def export_ab_mels(self, mel_i: torch.FloatTensor, pr: dict, i: int, prepend, iteration, dirname: str):
        # log results in a-b test path
        # for a-b test, write audio
        path = os.path.join('mels', prepend, dirname, os.path.splitext(os.path.split(pr['audiopath'][i])[-1])[0]+'.pt')
        os.makedirs(os.path.join(self.abtest_path, os.path.split(path)[0]), exist_ok=True)
        safe_write(
            mel_i,
            os.path.join(self.abtest_path, path),
        )
        # add line to ab-test list (with information on transcript, speaker, source audiofile, etc)
        with open(self.abtest_mellist_path, 'a') as f:
            line = [iteration, path, pr['sr'][i], pr['text_raw'][i], pr['spkrname'][i]]
            f.write('|'.join([str(x).replace("|", "") for x in line])+'\n')
    
    def export_ab_wavs(self, wav_i: torch.FloatTensor, pr: dict, i: int, prepend, iteration, dirname: str):
        # log results in a-b test path
        # for a-b test, write audio
        path = os.path.join('wavs', prepend, dirname, os.path.splitext(os.path.split(pr['audiopath'][i])[-1])[0]+'.wav')
        os.makedirs(os.path.join(self.abtest_path, os.path.split(path)[0]), exist_ok=True)
        write_wav(
            wav_i,
            os.path.join(self.abtest_path, path),
            self.sr,
        )
        # and add line to filelist (with information on transcript, speaker, source audiofile, etc)
        with open(self.abtest_wavlist_path, 'a') as f:
            line = [iteration, path, pr['sr'][i], pr['text_raw'][i], pr['spkrname'][i]]
            f.write('|'.join([str(x).replace("|", "") for x in line])+'\n')
    
    def tensor_log_spectrogram(self, iteration, spect, mel_lens, pr, prepend='NA', name='mel', range_=None):# [B, mel_T, n_mel]
        for i in range(min(self.n_plots, spect.shape[0])):
            mel_len = mel_lens.view(-1)[i].item()
            mel_i = spect[i, :mel_len, :].data.float().cpu()
            self.add_image(
                f"{prepend}_{i}/{name}",
                plot_spectrogram_to_numpy(
                    mel_i.transpose(0, 1).numpy(),
                    clim=(self.log_min_val, self.log_max_val) if range_ is None else range_,
                    title=f"{pr['spkrname'][i]}|{os.path.split(pr['audiopath'][i])[-1]}",
                ),
                iteration,
                dataformats='HWC',
            )
            
            if self.abtest_write_latest:
                self.export_ab_mels(mel_i, pr, i, prepend, iteration, 'latest')
            if iteration%self.abtest_write_interval==0:
                self.export_ab_mels(mel_i, pr, i, prepend, iteration, str(iteration))
    
    def tensor_log_time(self, iteration, gt_t, pr_t, gt_t_noisy, prepend='', name='', spkrname=None, audiopath=None, range_=None):# [B, mel_T, 1]
        gt_t, pr_t, gt_t_noisy = pad_same_length(gt_t, pr_t, gt_t_noisy)
        t = gt_t if gt_t is not None else pr_t
        for i in range(min(self.n_plots, gt_t.shape[0] if gt_t is not None else pr_t.shape[0])):
            non_zero = t[i].detach().abs().sum(1)!=0.0 # [B, wav_T, 1]
            nan_tensor = torch.tensor(float('nan'), device=t.device, dtype=t.dtype)
            
            try:
                self.add_image(
                    f"{prepend}_{i}/{name}",
                    plot_time_series_to_numpy(
                        gt_t      [i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if gt_t       is not None else None,
                        pr_t      [i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if pr_t       is not None else None,
                        gt_t_noisy[i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if gt_t_noisy is not None else None,
                        ymin=t[i].min().item(), ymax=t[i].max().item(),
                        title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}',
                        xlabel=f"Time (Green target, Red predicted{', Blue noisy input' if gt_t_noisy is not None else ''})",
                        ylabel=f"{name}",
                        color_a='green',
                        color_b='red',
                        color_c='blue',
                        label_a='target',
                        label_b='pred target',
                        label_c='noisy input',
                    ),
                    iteration,
                    dataformats='HWC',
                )
            except ValueError as ex:
                print("Failed to log", name)
                if gt_t is not None:
                    print("gt_t.shape =", gt_t.shape)
                if pr_t is not None:
                    print("pr_t.shape =", pr_t.shape)
                if gt_t_noisy is not None:
                    print("gt_t_noisy.shape =", gt_t_noisy.shape)
                raise ValueError(ex)
    
    def tensor_log_pitch(self, iteration, gt_f0, pr_f0, gt_f0_noisy, lens, prepend='NA', name='Log Pitch', spkrname=None, audiopath=None, range_=None):# [B, mel_T, 1]
        for i in range(min(self.n_plots, gt_f0.shape[0])):
            len    = lens[i].item()
           #voiced     = gt_f0[i, :len]!=0.0
           #nan_tensor = torch.tensor(float('nan'), device=gt_f0.device, dtype=gt_f0.dtype)
            
            try:
                self.add_image(
                    f"{prepend}_{i}/{name}",
                    plot_time_series_to_numpy(
                        gt_f0      [i, :len, 0].data.float().cpu().numpy() if gt_f0       is not None else None,
                        pr_f0      [i, :len, 0].data.float().cpu().numpy() if pr_f0       is not None else None,
                        gt_f0_noisy[i, :len, 0].data.float().cpu().numpy() if gt_f0_noisy is not None else None,
                        ymin=gt_f0[i].min().item(), ymax=gt_f0[i].max().item(),
                        title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}',
                        xlabel=f"Frames (Green target, Red predicted{', Blue noisy input' if gt_f0_noisy is not None else ''})",
                        ylabel=f"{name}",
                        color_a='green',
                        color_b='red',
                        color_c='blue',
                        label_a='target',
                        label_b='pred target',
                        label_c='noisy input',
                    ),
                    iteration,
                    dataformats='HWC',
                )
            except ValueError as ex:
                print("Failed to log", name)
                print("lens =", lens.tolist())
                if gt_f0 is not None:
                    print("gt_f0.shape =", gt_f0.shape)
                if pr_f0 is not None:
                    print("gt_f0.shape =", pr_f0.shape)
                if gt_f0_noisy is not None:
                    print("gt_f0.shape =", gt_f0_noisy.shape)
                raise ValueError(ex)
            
    def tensor_log_voiced(self, iteration, gt_vo, pr_vo, gt_vo_noisy, lens, prepend='NA', name='Voiced', spkrname=None, audiopath=None, range_=None):# [B, mel_T, 1]
        for i in range(min(self.n_plots, gt_vo.shape[0])):
            length = lens[i].item()
            
            try:
                self.add_image(
                    f"{prepend}_{i}/{name}",
                    plot_time_series_to_numpy(
                        gt_vo      [i, :length, 0].data.float().cpu().numpy() if gt_vo       is not None else None,
                        pr_vo      [i, :length, 0].data.float().cpu().numpy() if pr_vo       is not None else None,
                        gt_vo_noisy[i, :length, 0].data.float().cpu().numpy() if gt_vo_noisy is not None else None,
                        ymin=gt_vo[i].min().item(), ymax=gt_vo[i].max().item(),
                        title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}',
                        xlabel=f"Frames (Green target, Red predicted{', Blue noisy input' if gt_vo_noisy is not None else ''})",
                        ylabel=f"{name}",
                        color_a='green',
                        color_b='red',
                        color_c='blue',
                        label_a='target',
                        label_b='pred target',
                        label_c='noisy input',
                    ),
                    iteration,
                    dataformats='HWC',
                )
            except ValueError as ex:
                print("Failed to log", name)
                print("lens =", lens.tolist())
                if gt_vo is not None:
                    print("gt_vo.shape =", gt_vo.shape)
                if pr_vo is not None:
                    print("pr_vo.shape =", pr_vo.shape)
                if gt_vo_noisy is not None:
                    print("gt_vo_noisy.shape =", gt_vo_noisy.shape)
                raise ValueError(ex)
    
    def tensor_log_alignment(self, iteration, alignment, mel_lens, txt_lens, prepend='NA', name='alignment', spkrname=None, audiopath=None, text_symbols=None):# [B, mel_T, txt_T]
        mel_lens = mel_lens//round(mel_lens.max().item()/alignment.shape[2])
        for i in range(min(self.n_plots, alignment.shape[0])):
            mel_len = mel_lens[i].item()
            txt_len = txt_lens[i].item()
            symbols = text_symbols[i] if text_symbols is not None else None
            title = f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}'
            
            self.add_image(
                f"{prepend}_{i}/{name}",
                plot_alignment_to_numpy(
                    alignment[i, :txt_len, :mel_len].data.float().cpu().numpy(),
                    text_symbols=symbols, title=title
                ),
                iteration,
                dataformats='HWC'
            )
    
    def tensor_log_audio(self, iteration, wav, wav_lens, pr, prepend='', name=''):# [B, wav_T, 1]
        # log results in tensorboard
        for i in range(min(self.n_plots, wav.shape[0])):
            wav_len = wav_lens.view(-1)[i].item()
            wav_i = wav[i].data.squeeze().cpu()[:wav_len].unsqueeze(0)
            assert wav_i.dim() == 2, f"audio tensor has {wav_i.dim()} dims, expected 2 of shape [1, wav_T]. {wav_i.shape}"
            assert wav_i.shape[0] == 1, f"audio tensor has batch_size of {wav_i.shape[0]}, expected 1 for sliced item."
            assert wav_i.shape[1] > 32# check audio length is reasonable
            self.add_audio(
                f"{prepend}_{i}/{name}",
                wav_i,
                iteration,
                sample_rate=self.sr,
            )
            
            if self.abtest_write_latest:
                self.export_ab_wavs(wav_i, pr, i, prepend, iteration, 'latest')
            if iteration%self.abtest_write_interval==0:
                self.export_ab_wavs(wav_i, pr, i, prepend, iteration, str(iteration))
    
    def log_text_batch(self, iteration, text, prepend='', name=''):
        for i, t in enumerate(text):
            self.add_text(
                f"{prepend}_{i}/{name}",
                t,
                iteration,
            )
    
    def plot_pr_dict(self, pr, iteration, prepend=''):
        """Plot image/audio/text features from batch/batch dict"""
        pr_normal    = {}
        pr_same_spkr = {}
        pr_diff_spkr = {}
        for k, v in pr.items():
            if k.endswith('_same_spkr'):
                pr_same_spkr[k.replace('_same_spkr', '')] = v
            elif k.endswith('_diff_spkr'):
                pr_diff_spkr[k.replace('_diff_spkr', '')] = v
            else:
                pr_normal[k] = v
        
        self._plot_pr_dict(pr_normal, iteration, prepend=prepend, append='')
        if pr_same_spkr:
            self._plot_pr_dict(pr_same_spkr, iteration, prepend=prepend, append='_same_spkr')
        if pr_diff_spkr:
            self._plot_pr_dict(pr_diff_spkr, iteration, prepend=prepend, append='_diff_spkr')
    
    def _plot_pr_dict(self, pr, iteration, prepend='', append=''):
        audiopath = pr['audiopath']
        spkrname  = pr['spkrname']
        for k, v in pr.items():# find common plottable keys e.g 'mel','wav','f0' in keys
            if '_lens' in k:
                continue# skip 'mel_lens', 'wav_lens', etc
            name = k+append
            
            # Normal plots that only require the current key
            if any(plot_k in k for plot_k in self.keys_to_plot['spec']): # taken from config file
                mlens = pr['pr_mel_lens'] if 'pr_' in k and 'pr_mel_lens' in pr else pr['mel_lens']
                self.tensor_log_spectrogram(iteration, v.float(), mlens, pr, prepend=prepend, name=name, range_=None)
            
            if any(plot_k in k for plot_k in self.keys_to_plot['algn']): # taken from config file
                self.tensor_log_alignment(iteration, v.float(), pr['mel_lens'], pr['txt_lens'], text_symbols=pr['text_symbols'],
                                          prepend=prepend, name=name, spkrname=spkrname, audiopath=audiopath)
            
            if any(plot_k in k for plot_k in self.keys_to_plot['audio']) and 'wav_lens' in pr: # taken from config file
                wlens = pr['pr_wav_lens'] if 'pr_' in k and 'pr_wav_lens' in pr else pr['wav_lens']
                self.tensor_log_audio(iteration, v.float(), wlens, pr, prepend=prepend, name=name)
            
            if k == 'text':
                self.log_text_batch(iteration, v, prepend=prepend, name=name)
        
        # plot stuff that could require multiple keys on the same plot
        already_plotted = set()
        for k, v in pr.items():# find common plottable keys e.g 'mel','wav','f0' in keys
            if '_lens' in k:
                continue# skip 'mel_lens', 'wav_lens', etc
            if k in already_plotted:
                continue# and don't plot anything that has already been plotted.
            
            lens = pr['txt_lens'].squeeze() if any(x in k for x in ['gt_hc_','gt_sc_','_dur', '_logdur']) else pr['mel_lens'].squeeze()
            
            # Plots that might require the gt key and batch key
            gt_k, pr_k = get_pr_and_gt_versions(pr, k)
            name = k if (gt_k is None or pr_k is None) else k.replace('pr_', '').replace('gt_', '')
            name = name + append
            if any(plot_k in k for plot_k in self.keys_to_plot['time']) and (gt_k is not None and pr_k is not None): # taken from config file
                self.tensor_log_time(iteration,
                                     pr[gt_k].float() if gt_k is not None else None,
                                     pr[pr_k].float() if pr_k is not None else None,
                                     None,
                                     prepend=prepend, name=name, spkrname=spkrname, audiopath=audiopath, range_=None)
                if gt_k is not None: already_plotted.add(gt_k)
                if pr_k is not None: already_plotted.add(pr_k)
            
            if any(plot_k in k for plot_k in self.keys_to_plot['logpitch']) and (gt_k is not None and pr_k is not None): # taken from config file
                self.tensor_log_pitch(iteration,
                                      pr[gt_k].float() if gt_k is not None else None,
                                      pr[pr_k].float() if pr_k is not None else None,
                                      None, lens=lens,
                                      prepend=prepend, name=name, spkrname=spkrname, audiopath=audiopath, range_=None)
                if gt_k is not None: already_plotted.add(gt_k)
                if pr_k is not None: already_plotted.add(pr_k)
            
            if any(plot_k in k for plot_k in self.keys_to_plot['voiced']) and (gt_k is not None and pr_k is not None): # taken from config file
                self.tensor_log_voiced(iteration,
                                       pr[gt_k].float() if gt_k is not None else None,
                                       pr[pr_k].float() if pr_k is not None else None,
                                       None, lens=lens,
                                       prepend=prepend, name=name, spkrname=spkrname, audiopath=audiopath, range_=None)
                if gt_k is not None: already_plotted.add(gt_k)
                if pr_k is not None: already_plotted.add(pr_k)
    
    def plot_loss_dict(self, loss_dict_mean, iteration, prepend='', window_size=1):
        if window_size == 1:
            for loss_name, reduced_loss in loss_dict_mean.items():
                self.add_scalar(f"{prepend}/{loss_name}", reduced_loss, iteration)
        elif window_size > 1:
            
            for loss_name, reduced_loss in loss_dict_mean.items():
                k = f"{prepend}/{loss_name}"
                if k in self.running_loss_dict_mean:
                    self.running_loss_dict_mean[k].append(reduced_loss)
                else:
                    self.running_loss_dict_mean[k] = [reduced_loss, ]
                    self.running_loss_dict_start_iteration[k] = iteration
            
            for k, reduced_loss_list in self.running_loss_dict_mean.items():
                if len(reduced_loss_list) >= window_size:
                    reduced_loss_avg = sum(reduced_loss_list)/len(reduced_loss_list)
                    middle_iter = (iteration + self.running_loss_dict_start_iteration[k])//2 # if window spans from iter 0 to iter 10, it should log the average at iter 5
                    self.add_scalar(k, reduced_loss_avg, middle_iter)
                    self.running_loss_dict_mean[k] = []
            self.running_loss_dict_start_iteration.update({f"{prepend}/{loss_name}": iteration for loss_name, reduced_loss in loss_dict_mean.items()})
        else:
            raise ValueError(f'got window_size of {window_size}, expected int of 1 or greater')
     
    def log_parameters(self, model, iteration):# 'model.generator.speaker_encoder' -> 'generator.speaker_encoder/weight'
        for tag, value in model.named_parameters():
            self.add_histogram(
                tag.replace('generator.'    , 'generator_'    )
                   .replace('discriminator.', 'discriminator_')
                   .replace('.'             , '/'             ),
                value.data.cpu().numpy(), iteration)
    
    def log_speaker_encoder_projection(self, model):
        # plot speaker embed
        for k, v in model.state_dict().items():
            if '.spkr_embedding.weight' in k:
                # select embed vectors that have speakers attached
                spkr_ids = [speaker_id for speaker_name, speaker_id, dataset, source, source_type, duration in model.spkrlist]
                spkr_mask = torch.tensor([x in spkr_ids for x in range(v.shape[0])], dtype=torch.bool)
                embed = v[spkr_mask, :]
                
                metadata = [speaker_name for speaker_name, speaker_id, dataset, source, source_type, duration in model.spkrlist]
                self.add_embedding(
                    embed,# [B, D]
                    metadata,# [str,]*B
                    global_step=model.iteration,
                    tag=k,
                )
            if '.text_embedding.weight' in k:
                # select embed vectors that have speakers attached
                embed = v
                metadata = [symbol for symbol, text_id in model.textlist]
                
                self.add_embedding(
                    embed,# [B, D]
                    metadata,# [str,]*B
                    global_step=model.iteration,
                    tag=k,
                )
        
        # delete old embeds (ignoring oldest and newest entries)
        
        # update 'projector_config.pbtxt' to remove entries that no longer exist
        
        pass
    
    @torch.no_grad()
    def add_missing_metrics(self, pr):
        """Add pr_wav, pr_mel, ctc_loss and other metrics to batch if possible."""
        try: torch.cuda.empty_cache()
        except: pass
        
        if ('pr_mel' not in pr and
            'spkr_ids'   in pr and
            'text_ids'   in pr):
            if (is_feat_in_dict('dur'      , pr) and
                is_feat_in_dict('hc_nrg'   , pr) and
                is_feat_in_dict('hc_svo'   , pr) and
                is_feat_in_dict('hc_logsf0', pr)):
                if not getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder = maybe_load_model(**self.prosody_decoder_config)
                if getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder.cuda()
                    pr = self.prosody_decoder.infer_mel_from_chars(pr)
            elif (is_feat_in_dict('dur'   , pr) and
                  is_feat_in_dict('nrg'   , pr) and
                  is_feat_in_dict('svo'   , pr) and
                  is_feat_in_dict('logsf0', pr)):
                if not getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder = maybe_load_model(**self.prosody_decoder_config)
                if getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder.cuda()
                    pr = self.prosody_decoder.infer_mel_from_frames_with_dur(pr)
            elif (not is_feat_in_dict('dur', pr) and
                      is_feat_in_dict('nrg', pr) and
                      is_feat_in_dict('svo', pr) and
                      is_feat_in_dict('logsf0', pr)):
                if not getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder = maybe_load_model(**self.prosody_decoder_config)
                if getattr(self, 'prosody_decoder', None):
                    self.prosody_decoder.cuda()
                    pr = self.prosody_decoder.infer_mel_from_frames_without_dur(pr)
        
        if 'pr_mel' in pr and 'gt_mel' in pr:
            if not getattr(self, 'spkr_classifier', None):
                self.spkr_classifier = maybe_load_model(**self.spkr_classifier_config)
            if getattr(self, 'spkr_classifier', None):
                self.spkr_classifier.cuda()
                if pr['gt_mel'].shape[1] != pr['pr_mel'].shape[1]:
                    pr['loss_dict']['gt_pr_spkr_cosdist'] = 1.0 - self.spkr_classifier.get_similarity_global(
                        pr['gt_mel'  ], pr['pr_mel'],
                        pr['mel_lens'], get_matched_lens(pr['pr_mel'], [pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]),
                    ).mean().item()
                else:
                    pr['loss_dict']['gt_pr_spkr_cosdist'] = 1.0 - self.spkr_classifier.get_similarity_local(
                        pr['gt_mel'], pr['pr_mel'], pr['mel_lens']).mean().item()
        
        if 'pr_mel_diff_spkr' in pr:
            if 'gt_mel_diff_spkr' in pr:
                if not getattr(self, 'spkr_classifier', None):
                    self.spkr_classifier = maybe_load_model(**self.spkr_classifier_config)
                if getattr(self, 'spkr_classifier', None):
                    self.spkr_classifier.cuda()
                    pr['loss_dict']['diff_spkr_cosdist'] = 1.0 - self.spkr_classifier.get_similarity_global(
                        pr['gt_mel_diff_spkr'],
                        pr['pr_mel_diff_spkr'],
                        get_matched_lens(pr['gt_mel_diff_spkr'], [pr.get('mel_lens_diff_spkr', None), pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]),
                        get_matched_lens(pr['pr_mel_diff_spkr'], [pr.get('mel_lens_diff_spkr', None), pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]),
                    ).mean().item()
            else:
                if not getattr(self, 'spkr_classifier', None):
                    self.spkr_classifier = maybe_load_model(**self.spkr_classifier_config)
                if getattr(self, 'spkr_classifier', None):
                    self.spkr_classifier.cuda()
                    pr['loss_dict']['converted_spkr_cosdist'] = 1.0 - self.spkr_classifier.get_similarity_local(
                        pr['gt_mel'], pr['pr_mel_diff_spkr'], get_matched_lens(pr['gt_mel'], [pr.get('pr_mel_lens', None), pr.get('mel_lens', None)])
                    ).mean().item()
        
        if 'pr_mel' in pr and ('mel_lens' in pr or 'pr_mel_lens' in pr):
            if ('text_ids' in pr and
                'txt_CTC' not in pr.get('loss_dict', {})):
                if not getattr(self, 'text_classifier', None):
                    self.text_classifier = maybe_load_model(**self.text_classifier_config)
                if getattr(self, 'text_classifier', None):
                    self.text_classifier.cuda()
                    pr = self.text_classifier.infer_pr(pr)
            
            if 'spkr_ids' in pr:
                if 'pr_wav' not in pr: # pr_mel -> pr_wav
                    if not getattr(self, 'vocoder', None):
                        self.vocoder = maybe_load_model(**self.vocoder_config)
                    if getattr(self, 'vocoder', None):
                        self.vocoder.cuda()
                        pr['pr_wav'], pr['pr_wav_lens'] = self.vocoder.infer_specific(pr['pr_mel'], get_matched_lens(pr['pr_mel'], [pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]))
                        
                        if 'gt_wav' not in pr and 'gt_mel' in pr and self.has_not_written_gt_wav:
                            pr['gt_wav_rec'], pr['wav_lens'] = self.vocoder.infer_specific(pr['gt_mel'], get_matched_lens(pr['gt_mel'], [pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]))
                            self.has_not_written_gt_wav = False
                        else:
                            pr['wav_lens'] = pr['pr_wav_lens']
        
        if 'pr_mel_diff_spkr' in pr and 'mel_lens_diff_spkr' in pr:
            if 'spkr_ids' in pr:
                if 'pr_wav_diff_spkr' not in pr: # pr_mel -> pr_wav
                    if not getattr(self, 'vocoder', None):
                        self.vocoder = maybe_load_model(**self.vocoder_config)
                    if getattr(self, 'vocoder', None):
                        self.vocoder.cuda()
                        pr['pr_wav_diff_spkr'], pr['wav_lens_diff_spkr'] = self.vocoder.infer_specific(
                            pr['pr_mel_diff_spkr'],
                            get_matched_lens(
                                pr['gt_mel'],
                                [pr.get('mel_lens_diff_spkr', None), pr.get('pr_mel_lens', None), pr.get('mel_lens', None)]
                            )
                        )
        return pr
    
    def terminal_print(self, model, pr: dict,
                       loss_weights: dict, learning_rate: Optional[float], grad_norm: Optional[float], loss_scale: Optional[float],
                       start_model_time: float, start_data_time: float, prepend='train', eval_prog: str=''):
        if model.iteration%self.terminal_config['print_interval'] == 0:
            prntlist = [f'{prepend:14}', ]
            if self.terminal_config['show_eval_progress'] and eval_prog:
                prntlist.append(eval_prog)
            if self.terminal_config['show_iteration']:
                prntlist.append(f'{model.iteration:>7}')
            if self.terminal_config['show_total_loss'] and ('loss_g_reduced' in pr or 'loss_d_reduced' in pr):
                total_loss = pr.get('loss_g_reduced', 0.0) + pr.get('loss_d_reduced', 0.0)
                prntlist.append(f'{total_loss:.2f}loss')
            if self.terminal_config['show_grad_norm'] and grad_norm is not None:
                prntlist.append(f'{grad_norm:.2f}gnorm')
            if self.terminal_config['show_iter_time']:
                prntlist.append(f'{time.time()-start_model_time:.2f}s/iter')
            if self.terminal_config['show_time_per_file']:
                total_batch_size = model.h['dataloader_config']['batch_size']*model.h['n_rank']
                prntlist.append(f'{(time.time()-start_model_time)/total_batch_size:.2f}s/file')
            if self.terminal_config['show_dataloading_time']:
                prntlist.append(f'{start_model_time-start_data_time:.2f}IO s/iter')
            if self.terminal_config['show_learning_rate'] and learning_rate is not None:
                prntlist.append(f'{learning_rate:.2e}lr'.replace('-0', '-').replace('+0', '+'))
            if loss_scale is not None and self.amp_level > 0:
                prntlist.append(f'{loss_scale:.0f}LS')
            if self.terminal_config['show_top1_att_strength']:
                if 'loss_dict' in pr and 'top1_avg_prob' in pr['loss_dict']:
                    prntlist.append(f"{pr['loss_dict']['top1_avg_prob']:.2f} 1avgp")
            if self.terminal_config['show_top2_att_strength']:
                if 'loss_dict' in pr and 'top2_avg_prob' in pr['loss_dict']:
                    prntlist.append(f"{pr['loss_dict']['top2_avg_prob']:.2f} 2avgp")
            print("|".join(prntlist))
    
    def log_training(self, pr, model, prepend='train', validation_inverval=None):
        # get average loss values across GPUs and plot to tensorboard
        loss_dict_reduced = pr['loss_dict'].copy()
        if 'loss_g_reduced' in pr:
            loss_dict_reduced["loss_g_reduced"] = pr['loss_g_reduced']
        if 'loss_d_reduced' in pr:
            loss_dict_reduced["loss_d_reduced"] = pr['loss_d_reduced']
        
        # plot scalars to line graphs (maybe with multiple time-scales)
        steps = [['', model.iteration], ]
        if self.enable_milliepoch_plots:
            steps.append(['_milliepoch', int(round(model.epoch*1000.0))])
        for prepend_label, global_step in steps:# get average loss values across GPUs and plot to tensorboard
            self.plot_loss_dict(loss_dict_reduced, global_step, prepend=prepend+prepend_label, window_size=self.training_window_size)
        
        # update losses for each path
        self.loss_dict_path.update(pr['loss_dict_path'])
        
        if validation_inverval is not None and model.iteration%validation_inverval==0:
            print("Logging training results...")
            self.plot_pr_dict(pr, model.iteration, prepend=prepend)# extract common keys from batch dict and plot them
    
    def log_evaluation(self, pr, model, prepend=''):# get batch from first n_dynamic_plots items in validation
        print(f"Logging {prepend} results...")
        
        pr = self.add_missing_metrics(pr)
        
        # image/audio/text plots for tensorboard
        self.plot_pr_dict(pr, model.iteration, prepend=prepend)
            
        # plot parameters
        self.log_parameters(model, model.iteration)
        
        # plot text / spkr embed projections
        self.log_speaker_encoder_projection(model)
        
        # plot scalars to line graphs (maybe with multiple time-scales)
        steps = [['', model.iteration], ]
        if self.enable_milliepoch_plots:
            steps.append(['_milliepoch', int(round(model.epoch*1000.0))])
        for prepend_label, global_step in steps:# get average loss values across GPUs and plot to tensorboard
            if 'loss_dict' in pr:
                self.plot_loss_dict(pr['loss_dict'], global_step, prepend=prepend+prepend_label)
            if 'loss_g_reduced' in pr:
                self.add_scalar(f"{prepend+prepend_label}/loss_g_reduced", pr['loss_g_reduced'], global_step)
            if 'loss_d_reduced' in pr:
                self.add_scalar(f"{prepend+prepend_label}/loss_d_reduced", pr['loss_d_reduced'], global_step)
        
        # update losses for each path
        if 'loss_dict_path' in pr:
            self.loss_dict_path.update(pr['loss_dict_path'])
        
        #self.flush() # will force-stop training to write to disk, not really needed since async writing works well already.
        print(f"Finished logging {prepend}")

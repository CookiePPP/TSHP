import logging
import math
from copy import deepcopy

from TSHP.modules.train_optim import MovingDictAverage
from TSHP.utils.misc_utils import zip_equal

from TSHP.utils.dataset.dataset import Collate

from TSHP.utils.modules.core import dist_add
from TSHP.utils.saving.utils import safe_write

log = logging.getLogger('rich')
import TSHP.utils.warnings as w

import os
import random
import time

import torch
from torch import Tensor
from typing import Optional, Dict, List, Union

from TSHP.utils.dataset.audio.stft import mag_to_log
from TSHP.utils.logging.plotting import plot_alignment_to_numpy, plot_time_series_to_numpy, plot_spectrogram_to_numpy
from TSHP.utils.modules.utils import Fpad

def maybe_load_model(ref, path):
    if not (ref and path):
        return None
    # import the model's code
    name = "Model"
    modelmodule = getattr(
        __import__(f'TSHP.models.{ref}.model', fromlist=[name]), name)
    # load checkpoint
    model, *_ = modelmodule.load_model(path, train=False)
    return model

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

def de_ambiguise_lens(t: Tensor, lens_list: List, allow_return_none: bool =False):
    t_mask = t.abs().sum(dim=2, keepdim=True).ne(0)
    t_lens = t_mask.sum(1, True)
    for lens in lens_list:
        if lens.to(t_lens).view(-1).tolist() == t_lens.view(-1).tolist():
            return lens
    
    if allow_return_none:
        return
    else:
        raise RuntimeWarning(f"found no lens that match input tensor. input tensor len is {t.shape[1]}")

def get_matched_lens(tensor: Tensor, lens_list: Union[List[Tensor], Dict[str, Tensor]], key: Optional[str]=None, allow_return_none: bool=False):
    assert len(lens_list) > 0, f'lens_list has len of {len(lens_list)}'
    if isinstance(lens_list, (list, tuple)):
        matching_lens = [t for t in lens_list if t is not None and tensor.shape[1] == t.max().item()]
        if len(matching_lens) == 1:
            return matching_lens[0]
        else:
            return de_ambiguise_lens(tensor, matching_lens if len(matching_lens) else lens_list, allow_return_none=allow_return_none)
    
    elif isinstance(lens_list, dict):
        if key is not None:
            # check matching arpa type
            if key.startswith('p_'):
                matching_lens = [t for k, t in lens_list.items() if t is not None and tensor.shape[1] == t.max().item() and k.startswith('p_')]
                if len(matching_lens) == 1:
                    return matching_lens[0]
                else:
                    return de_ambiguise_lens(tensor, matching_lens if len(matching_lens) else list(lens_list.values()), allow_return_none=allow_return_none)
            elif key.startswith('g_'):
                matching_lens = [t for k, t in lens_list.items() if t is not None and tensor.shape[1] == t.max().item() and k.startswith('g_')]
                if len(matching_lens) == 1:
                    return matching_lens[0]
                else:
                    return de_ambiguise_lens(tensor, matching_lens if len(matching_lens) else list(lens_list.values()), allow_return_none=allow_return_none)
            else:
                matching_lens = [t for k, t in lens_list.items() if t is not None and tensor.shape[1] == t.max().item() and not (k.startswith('p_') or k.startswith('g_'))]
                if len(matching_lens) == 1:
                    return matching_lens[0]
                else:
                    return de_ambiguise_lens(tensor, matching_lens if len(matching_lens) else list(lens_list.values()), allow_return_none=allow_return_none)
        else:
            # check all types (starting with non-arpa)
            matching_lens = [t for k, t in lens_list.items() if t is not None and tensor.shape[1] == t.max().item()]
            return matching_lens[0]
    
    if allow_return_none:
        return
    else:
        raise RuntimeWarning(f"found no lens that match input tensor. input tensor len is {tensor.shape[1]}")


if __name__ == '__main__':
    # test get_matched_lens()
    
    # test with obvious lengths # TODO
    t = torch.randn(2, 640, 8)
    lens_list = [
        torch.tensor([640, 320]),
        torch.tensor([480, 240]),
    ]
    lens = get_matched_lens(t, lens_list)
    assert lens[0] == 640
    lens = get_matched_lens(t, lens_list[::-1])
    assert lens[0] == 640
    
    # test with ambiguous lengths but zero-masked padding # TODO
    t = torch.randn(2, 640, 8)
    t[1, 240:].fill_(0.0)
    lens_list = [
        torch.tensor([640, 320]),
        torch.tensor([640, 240]),
    ]
    lens = get_matched_lens(t, lens_list)
    assert lens[1] == 240
    lens = get_matched_lens(t, lens_list[::-1])
    assert lens[1] == 240
    
    # test with ambiguous lengths, but key uses 'p_' or 'g_' # TODO
    t = torch.randn(3, 640, 8)
    lens_list = {
          'lens': torch.tensor([640, 180, 240]), # [?, g, p]
        'g_lens': torch.tensor([640, 180, 180]), # [g, g, g]
        'p_lens': torch.tensor([640, 240, 240]), # [p, p, p]
    }
    lens = get_matched_lens(t, lens_list, key='p_')
    assert lens.tolist() == [640, 240, 240]
    
    # test with ambiguous lengths, but key doesn't use 'p_' or 'g_' # TODO
    t = torch.randn(3, 640, 8)
    lens_list = {
          'lens': torch.tensor([640, 180, 240]), # [?, g, p]
        'g_lens': torch.tensor([640, 180, 180]), # [g, g, g]
        'p_lens': torch.tensor([640, 240, 240]), # [p, p, p]
    }
    lens = get_matched_lens(t, lens_list, key='')
    assert lens.tolist() == [640, 180, 240]
    
    w.print1('get_matched_lens() Test Completed!')

def pad_same_length(*tensors, max_T=None):
    if max_T is None:
        max_T = max(t.shape[1] for t in tensors if t is not None)
    
    tensors_out = []
    for tensor in tensors:
        if tensor is not None:
            tensors_out.append(Fpad(tensor, (0, max_T-tensor.shape[1])))
        else:
            tensors_out.append(tensor)
    return tuple(tensors_out)


from scipy.io.wavfile import write
def write_wav(audio: torch.FloatTensor, path: str, sampling_rate: Union[int, float]):
    # scale audio for int16 output
    audio = (audio.float() * 2**15).squeeze().numpy().astype('int16')
    tmppath = f'{path}{random.randint(0, 9999)}.tmp'
    write(tmppath, sampling_rate, audio)# write to tmp path before moving so corruption to target path doesn't occur if the write stops part-way-through
    os.rename(tmppath, path)

class MockSummaryWriter:
    def __init__(self, log_dir, max_queue, flush_secs):
        assert isinstance(log_dir, str)
        assert isinstance(max_queue, int)
        assert max_queue > 0
        assert isinstance(flush_secs, int)
        assert flush_secs > 0
        
        self.plotted_scalar_tags = {}
        self.plotted_image_tags = {}
        self.plotted_audio_tags = {}
    
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        assert isinstance(tag, str)
        assert isinstance(scalar_value, (int, float, torch.Tensor)), f'{tag} is not float or Tensor'
        assert isinstance(global_step, int)
        assert global_step >= 0
        self.plotted_scalar_tags[tag] = scalar_value
        
    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        assert dataformats == 'HWC' # currently all code uses HWC so this should be enforced in future
        assert isinstance(tag, str)
        assert len(img_tensor.shape) == 3
        assert img_tensor.shape[2] == 3 # assert RGB image
        assert isinstance(global_step, int)
        assert global_step >= 0
        self.plotted_image_tags[tag] = img_tensor
    
    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        assert isinstance(tag, str)
        assert len(snd_tensor.shape) == 2
        assert snd_tensor.shape[0] == 1
        assert snd_tensor.max() <=  1.0
        assert snd_tensor.min() >= -1.0
        assert isinstance(global_step, int)
        assert global_step >= 0
        assert isinstance(sample_rate, int)
        self.plotted_audio_tags[tag] = snd_tensor


from torch.utils.tensorboard import SummaryWriter
class TensorBoardLogger:
    def __init__(self, run_path, tensorboardlogger_config: dict, h: dict, mock=False):
        
        # unit test stuff
        self.mock = mock
        self.tag_shapes = {} if mock else None # {tag: Tensor.shape, ...}
        SummaryWriterClass = MockSummaryWriter if self.mock else SummaryWriter
        
        # initialize SummaryWriter
        self.max_queue = tensorboardlogger_config['max_queue']
        self.flush_secs = tensorboardlogger_config['flush_secs']
        assert isinstance(run_path, str), 'run_path is not a string'
        assert run_path.strip() != '', 'run_path is empty'
        log_dir = os.path.join(run_path, f'logs_{str(random.randint(0, 1000000)).zfill(7)}') if tensorboardlogger_config.get('tuning', False) else os.path.join(run_path, 'logs')
        self.summarywriter = SummaryWriterClass(log_dir=log_dir, max_queue=self.max_queue, flush_secs=self.flush_secs)
        
        # print command for launching tensorboard
        mtitle = '.'.join(reversed(h["model_identity"].split(".")))
        print(f'cd "{os.path.split(run_path)[0]}"; tensorboard --logdir ./ --samples_per_plugin scalars=10000,images=100,audio=100 --bind_all --load_fast true --window_title \"{mtitle}\"')
        print(f'cd "{os.path.split(run_path)[0]}"; tensorboard --logdir ./ --samples_per_plugin scalars=10000,images=100,audio=100 --bind_all --load_fast true --window_title \"{mtitle}\"')
        
        # scalar params
        self.plot_misc = tensorboardlogger_config['plot_misc']
        self.plot_raw = tensorboardlogger_config['plot_raw']
        self.plot_expavg = tensorboardlogger_config['plot_expavg']
        self.plot_epochavg = tensorboardlogger_config['plot_epochavg']
        self.plot_best_expavg = tensorboardlogger_config['plot_best_expavg']
        self.plot_best_epochavg = tensorboardlogger_config['plot_best_epochavg']
        
        # media params
        self.sr = h['dataloader_config']['audio_config']['sr']
        
        clamp_val = float(h['dataloader_config']['stft_config']['clamp_val'])
        self.log_min_val = mag_to_log(clamp_val, 0.0, h['dataloader_config']['stft_config']['log_type'])
        self.log_max_val = mag_to_log(     60.0, 0.0, h['dataloader_config']['stft_config']['log_type'])
        
        self.keys_to_plot = tensorboardlogger_config['keys_to_plot']
        self.plot_width = tensorboardlogger_config['plot_width']
        self.plot_height = tensorboardlogger_config['plot_height']
    
    def _plot_scalar_dict(self, d, postpend, prepend, step):
        for k, v in d.items():
            if v is not None:
                self.summarywriter.add_scalar(f"{prepend}{postpend}/{k}", v, step)
    
    def _tensor_log_spectrogram(self, iteration, spect, mel_lens, pr, tag_list: List[str]=None, range_=None, pitch=None):# [B, mel_T, n_mel]
        for i in range(spect.shape[0]):
            mel_len = mel_lens.view(-1)[i].item()
            mel_i = spect[i, :mel_len, :].data.float().cpu()
            mel_i = mel_i.transpose(0, 1).numpy()
            
            f0_i = None
            if pitch is not None:
                f0_i = pitch[i, :mel_len, 0].data.float().cpu()
            
            tag = tag_list[i]
            clim = (self.log_min_val, self.log_max_val) if range_ is None else range_
            title = f"{pr['spkrname'][i]}|{os.path.split(pr['audiopath'][i])[-1]}"
            self.summarywriter.add_image(
                tag, plot_spectrogram_to_numpy(mel_i, clim=clim, title=title, logpitch=f0_i),
                iteration, dataformats='HWC'
            )
            if self.mock:
                self.tag_shapes[tag] = mel_i.shape
    
    def _tensor_log_time(self, iteration, gt_t, pr_t, gt_t_noisy, name='', tag_list: List[str]=None, spkrname=None, audiopath=None):# [B, mel_T, 1]
        gt_t, pr_t, gt_t_noisy = pad_same_length(gt_t, pr_t, gt_t_noisy)
        t = gt_t if gt_t is not None else pr_t
        for i in range(gt_t.shape[0] if gt_t is not None else pr_t.shape[0]):
            non_zero = t[i].detach().abs().sum(1)!=0.0 # [B, wav_T, 1]
            nan_tensor = torch.tensor(float('nan'), device=t.device, dtype=t.dtype)
            
            gt_ti   = gt_t      [i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if gt_t       is not None else None
            pr_ti   = pr_t      [i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if pr_t       is not None else None
            gt_ti_n = gt_t_noisy[i, :, 0].data.float().where(non_zero, nan_tensor).cpu().numpy() if gt_t_noisy is not None else None
            
            ymin=t[i].min().item()
            ymax=t[i].max().item()
            title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}'
            xlabel=f"Time (Green target, Red predicted{', Blue noisy input' if gt_t_noisy is not None else ''})"
            ylabel=f"{name}"
            color_a='green'
            color_b='red'
            color_c='blue'
            label_a='target'
            label_b='pred target'
            label_c='noisy input'
            tag = tag_list[i]
            
            assert math.isfinite(ymin), f'{name} has invalid min value of {ymin}'
            assert math.isfinite(ymax), f'{name} has invalid max value of {ymax}'
            
            self.summarywriter.add_image(
                tag,
                plot_time_series_to_numpy(
                    gt_ti, pr_ti, gt_ti_n,
                    ymin=ymin, ymax=ymax,
                    title=title, xlabel=xlabel, ylabel=ylabel,
                    color_a=color_a, color_b=color_b, color_c=color_c,
                    label_a=label_a, label_b=label_b, label_c=label_c,
                ), iteration, dataformats='HWC')
            if self.mock:
                self.tag_shapes[tag] = gt_ti.shape
    
    def _tensor_log_pitch(self, iteration, gt_f0, pr_f0, gt_f0_noisy, lens, name='Log Pitch', tag_list: List[str]=None, spkrname=None, audiopath=None, mask_zeros=False):# [B, mel_T, 1]
        for i in range(gt_f0.shape[0]):
            leni = lens[i].item()
            if mask_zeros:
                voiced     = gt_f0[i, :leni]!=0.0
                nan_tensor = torch.tensor(float('nan'), device=gt_f0.device, dtype=gt_f0.dtype)
                raise NotImplementedError
            
            gt_f0i   = gt_f0      [i, :leni, 0].data.float().cpu().numpy() if gt_f0       is not None else None
            pr_f0i   = pr_f0      [i, :leni, 0].data.float().cpu().numpy() if pr_f0       is not None else None
            gt_f0i_n = gt_f0_noisy[i, :leni, 0].data.float().cpu().numpy() if gt_f0_noisy is not None else None
            ymin=gt_f0[i].min().item()
            ymax=gt_f0[i].max().item()
            title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}'
            xlabel=f"Frames (Green target, Red predicted{', Blue noisy input' if gt_f0_noisy is not None else ''})"
            ylabel=f"{name}"
            color_a='green'
            color_b='red'
            color_c='blue'
            label_a='target'
            label_b='pred target'
            label_c='noisy input'
            tag = tag_list[i]
            
            self.summarywriter.add_image(
                tag,
                plot_time_series_to_numpy(
                    gt_f0i, pr_f0i, gt_f0i_n,
                    ymin=ymin, ymax=ymax,
                    title=title, xlabel=xlabel, ylabel=ylabel,
                    color_a=color_a, color_b=color_b, color_c=color_c,
                    label_a=label_a, label_b=label_b, label_c=label_c,
                ), iteration, dataformats='HWC')
            if self.mock:
                self.tag_shapes[tag] = gt_f0i.shape
    
    def _tensor_log_voiced(self, iteration, gt_vo, pr_vo, gt_vo_noisy, lens, name='Voiced', tag_list: List[str]=None, spkrname=None, audiopath=None, range_=None):# [B, mel_T, 1]
        for i in range(gt_vo.shape[0]):
            leni = lens[i].item()
            
            gt_voi   = gt_vo      [i, :leni, 0].data.float().cpu().numpy() if gt_vo       is not None else None
            pr_voi   = pr_vo      [i, :leni, 0].data.float().cpu().numpy() if pr_vo       is not None else None
            gt_voi_n = gt_vo_noisy[i, :leni, 0].data.float().cpu().numpy() if gt_vo_noisy is not None else None
            ymin=gt_vo[i].min().item()
            ymax=gt_vo[i].max().item()
            title=f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}'
            xlabel=f"Frames (Green target, Red predicted{', Blue noisy input' if gt_vo_noisy is not None else ''})"
            ylabel=f"{name}"
            color_a='green'
            color_b='red'
            color_c='blue'
            label_a='target'
            label_b='pred target'
            label_c='noisy input'
            tag = tag_list[i]
            
            self.summarywriter.add_image(
                tag,
                plot_time_series_to_numpy(
                    gt_voi, pr_voi, gt_voi_n,
                    ymin=ymin, ymax=ymax,
                    title=title, xlabel=xlabel, ylabel=ylabel,
                    color_a=color_a, color_b=color_b, color_c=color_c,
                    label_a=label_a, label_b=label_b, label_c=label_c,
                ), iteration, dataformats='HWC')
            if self.mock:
                self.tag_shapes[tag] = gt_voi.shape
    
    def _tensor_log_alignment(self, iteration, alignment, mel_lens, txt_lens, tag_list: List[str]=None, spkrname=None, audiopath=None, text_symbols=None):# [B, mel_T, txt_T]
        mel_lens = mel_lens//round(mel_lens.max().item()/alignment.shape[2])
        for i in range(alignment.shape[0]):
            mel_len = mel_lens[i].item()
            txt_len = txt_lens[i].item()
            symbols = text_symbols[i] if text_symbols is not None else None
            title = f'{spkrname[i]}|{os.path.split(audiopath[i])[-1]}'
            
            alignmenti = alignment[i, :txt_len, :mel_len].data.float().cpu().numpy()
            
            tag = tag_list[i]
            self.summarywriter.add_image(
                tag,
                plot_alignment_to_numpy(
                    alignmenti, text_symbols=symbols, title=title
                ), iteration, dataformats='HWC')
            if self.mock:
                self.tag_shapes[tag] = alignmenti.shape
    
    def _tensor_log_audio(self, iteration, wav, wav_lens, tag_list: List[str]=None):# [B, wav_T, 1]
        # log results in tensorboard
        for i in range(wav.shape[0]):
            wav_len = wav_lens.view(-1)[i].item()
            wav_i = wav[i].data.view(-1).cpu()[:wav_len].unsqueeze(0)
            assert wav_i.dim() == 2, f"audio tensor has {wav_i.dim()} dims, expected 2 of shape [1, wav_T]. {wav_i.shape}"
            assert wav_i.shape[0] == 1, f"audio tensor has batch_size of {wav_i.shape[0]}, expected 1 for sliced item."
            assert wav_i.shape[1] > 32# check audio length is reasonable
            tag = tag_list[i]
            self.summarywriter.add_audio(
                tag,
                wav_i,
                iteration,
                sample_rate=self.sr,
            )
    
    def _plot_pr_dict(self, batch, iteration, prepend='', append='', use_name_tags=False):
        audiopaths = batch['audiopath']
        spkrname  = batch['spkrname']
        for k, v in batch.items():# find common plottable keys e.g 'mel','wav','f0' in keys
            if type(v) is not torch.Tensor:
                continue
            if '_lens' in k:
                continue# skip/don't plot "mel_lens", "wav_lens", etc
            name = k+append
            
            if use_name_tags:
                filenames = [os.path.split(audiopath)[-1] for audiopath in audiopaths]
                filebases = [os.path.splitext(filename)[0][:12] for filename in filenames]
                tag_list = [f"{prepend}_static_{filebases[i]}/{name}" for i in range(len(v))] # eg: 'train_static_00_01_44_Flu/gt_mel'
            else:
                tag_list = [f"{prepend}_{i}/{name}" for i in range(len(v))] # eg: 'train_0/gt_mel'
            
            # Normal plots that only require the current key
            if any(plot_k in k for plot_k in self.keys_to_plot['spec']): # taken from config file
                mlens = get_matched_lens(v, [t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt and 'mel' in kt])
                self._tensor_log_spectrogram(
                    iteration, v.float(), mlens, batch, tag_list=tag_list, range_=None, pitch=batch.get('gt_logsf0', None))
            
            if any(plot_k in k for plot_k in self.keys_to_plot['align']): # taken from config file
                try:
                    elens = batch['txt_lens']#get_matched_lens(v, {kt: t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt}, key=k)
                    dlens = batch['mel_lens']#get_matched_lens(v.transpose(1, 2), {kt: t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt}, key=k)
                    self._tensor_log_alignment(
                        iteration, v.float(), dlens, elens, tag_list=tag_list, text_symbols=batch['text_symbols'], spkrname=spkrname, audiopath=audiopaths)
                except RuntimeWarning as ex:
                    w.print4exc(ex)
            
            if any(plot_k in k for plot_k in self.keys_to_plot['audio']): # taken from config file
                wlens = get_matched_lens(v, [t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt and 'wav' in kt])
                self._tensor_log_audio(
                    iteration, v.float(), wlens, tag_list=tag_list)
        
        # plot stuff that could require multiple keys on the same plot
        already_plotted = set()
        for k, v in batch.items():# find common plottable keys e.g 'mel','wav','f0' in keys
            if type(v) is not torch.Tensor:
                continue
            if '_lens' in k:
                continue# skip/don't plot "mel_lens", "wav_lens", etc
            if k in already_plotted:
                continue# and don't plot anything that has already been plotted.
            
            #lens = batch['txt_lens'].view(-1) if any(x in k for x in ['gt_hc_', 'gt_sc_', '_dur', '_logdur']) else batch['mel_lens'].view(-1)
            
            # Plots that might require the gt key and batch key
            gt_k, pr_k = get_pr_and_gt_versions(batch, k)
            
            name = k if (gt_k is None or pr_k is None) else k.replace('pr_', '').replace('gt_', '')
            name = name + append
            
            if use_name_tags:
                filenames = [os.path.split(audiopath)[-1] for audiopath in audiopaths]
                filebases = [os.path.splitext(filename)[0][:12] for filename in filenames]
                tag_list = [f"{prepend}_static_{filebases[i]}/{name}" for i in range(len(v))] # eg: 'train_static_00_01_44_Flu/sf0'
            else:
                tag_list = [f"{prepend}_{i}/{name}" for i in range(len(v))] # eg: 'train_0/sf0'
            
            if any(plot_k in k for plot_k in self.keys_to_plot['time']): # taken from config file
                self._tensor_log_time(iteration,
                                     batch[gt_k].float() if gt_k is not None else None,
                                     batch[pr_k].float() if pr_k is not None else None,
                                     None, tag_list=tag_list,
                                     name=name, spkrname=spkrname, audiopath=audiopaths)
                if gt_k is not None: already_plotted.add(gt_k)
                if pr_k is not None: already_plotted.add(pr_k)
            
            try:
                if any(plot_k in k for plot_k in self.keys_to_plot['pitch']): # taken from config file
                    lens = get_matched_lens(v, {kt: t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt}, key=k)
                    self._tensor_log_pitch(iteration,
                                          batch[gt_k].float() if gt_k is not None else None,
                                          batch[pr_k].float() if pr_k is not None else None,
                                          None, lens=lens, tag_list=tag_list,
                                          name=name, spkrname=spkrname, audiopath=audiopaths)
                    if gt_k is not None: already_plotted.add(gt_k)
                    if pr_k is not None: already_plotted.add(pr_k)
                
                if any(plot_k in k for plot_k in self.keys_to_plot['voiced']): # taken from config file
                        lens = get_matched_lens(v, {kt: t for kt, t in batch.items() if type(t) is torch.Tensor and 'lens' in kt}, key=k)
                        self._tensor_log_voiced(iteration,
                                               batch[gt_k].float() if gt_k is not None else None,
                                               batch[pr_k].float() if pr_k is not None else None,
                                               None, lens=lens, tag_list=tag_list,
                                               name=name, spkrname=spkrname, audiopath=audiopaths, range_=None)
                        if gt_k is not None: already_plotted.add(gt_k)
                        if pr_k is not None: already_plotted.add(pr_k)
            except RuntimeWarning as ex:
                w.print4exc(ex)
    
    def plot_scalars(self, prepend, additional_scalars, loss_dict,
                     grad_norm_total, grad_norm_g, grad_norm_d,
                     log_loss_scale, learning_rate, batch_size,
                     current_time, iters, epoch, secpr,
                     model_time_elapsed, model_secpr_per_second,
                     io_time_elapsed, io_secpr_per_second,
                     full_time_elapsed, full_secpr_per_second,
                     expavg_loss_dict, epochavg_loss_dict,
                     bestexpavg_loss_dict, bestepochavg_loss_dict):
        misc_dict = {
            'grad_norm_total': grad_norm_total,
            'grad_norm_g': grad_norm_g,
            'grad_norm_d': grad_norm_d,
            'log_loss_scale': log_loss_scale,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'iters': iters,
            'epoch': epoch,
            'model_secpr_per_second': model_secpr_per_second,
            'io_secpr_per_second': io_secpr_per_second,
            'full_secpr_per_second': full_secpr_per_second,
        }
        secpr_int = int(round(secpr))
        if self.plot_misc:
            self._plot_scalar_dict(             misc_dict, postpend='_misc'    , prepend=prepend, step=secpr_int)
        if self.plot_raw:
            self._plot_scalar_dict(             loss_dict, postpend='_raw'     , prepend=prepend, step=secpr_int)
        if self.plot_expavg:
            self._plot_scalar_dict(      expavg_loss_dict, postpend='_expavg'  , prepend=prepend, step=secpr_int)
        if self.plot_epochavg:
            self._plot_scalar_dict(    epochavg_loss_dict, postpend='_epochavg', prepend=prepend, step=secpr_int)
        if self.plot_best_expavg:
            self._plot_scalar_dict(  bestexpavg_loss_dict, postpend='_best_expavg', prepend=prepend, step=secpr_int)
        if self.plot_best_epochavg:
            self._plot_scalar_dict(bestepochavg_loss_dict, postpend='_best_epochavg', prepend=prepend, step=secpr_int)
    
    def plot_media(self, prepend, batch, batch_dynamic_plots: Optional[Dict[str, Tensor]], batch_static_plots: Optional[Dict[str, Tensor]], batch_size, current_time, iters, epoch, secpr):
        secpr_int = int(round(secpr))
        if batch_dynamic_plots is not None:
            self._plot_pr_dict(batch_dynamic_plots, secpr_int, prepend=prepend)
        if batch_static_plots is not None:
            self._plot_pr_dict(batch_static_plots, secpr_int, prepend=prepend, use_name_tags=True)


if __name__ == '__main__':
    # test TensorBoardLogger
    w.setLevel('info')
    
    run_path = 'outdir/logdir'
    tensorboardlogger_config = {
        'max_queue': 120,
        'flush_secs': 120,
        'tuning': False,
        'plot_misc': True,
        'plot_raw': True,
        'plot_expavg': True,
        'plot_epochavg': True,
        'plot_best_expavg': True,
        'plot_best_epochavg': True,
        'keys_to_plot': {
            'spec': ['spec', 'mel'],
            'time': ['wav', 'vol', 'nrg', 'gate', 'dur', 'logdur'],
            'pitch': ['f0'],
            'voiced': ['vo', 'svo', 'sf0'],
            'align': ['alignment'],
            'audio': ['wav'],
        },
        'plot_width': 12.0,
        'plot_height': 4.0,
    }
    h = {
        "model_identity": 'mock.tacotron2',
        'dataloader_config': {
            'audio_config': {
                'sr': 44100,
            },
            'stft_config': {
                'clamp_val': 1e-5,
                'log_type': 'log',
            }
        },
        "metricmodule_config": {
            "n_static_plots": 10,
            "minutes_between_dynamic_plots": 0.0,
            "n_dynamic_plots": 10,
        }
    }
    
    # initialize TBLogger using recommended values (with mock SummaryWriter to not affect the filesystem)
    tblogger_base = TensorBoardLogger(run_path, tensorboardlogger_config, h, mock=True)
    
    # test with plotting scalars
    tblogger = deepcopy(tblogger_base)
    tblogger.plot_scalars(
        prepend='train',
        additional_scalars={
            "grad_norm_total": 2.0,
            "grad_norm_g": 1.0,
            "grad_norm_d": 1.0,
            "loss_scale": 2**5,
            "learning_rate": 1e-5,
        },
        loss_dict={'mel_MAE': 1.0},
        grad_norm_total=2.0,
        grad_norm_g=1.0,
        grad_norm_d=1.0,
        log_loss_scale=5.0,
        learning_rate=1e-5,
        batch_size=96,
        current_time=time.time(),
        iters = 100_000,
        epoch = 5.0,
        secpr = 800_000,
        model_time_elapsed = 1.0,
        model_secpr_per_second = 150.0,
        io_time_elapsed = 0.01,
        io_secpr_per_second = 15_000.0,
        full_time_elapsed = 1.01,
        full_secpr_per_second = 148.5,
        expavg_loss_dict = {},
        epochavg_loss_dict = {},
        bestexpavg_loss_dict = {},
        bestepochavg_loss_dict = {},
    )
    
    common_kwargs = {
        'batch_size': 64,
        'current_time': time.time(),
        'iters': 100_000,
        'epoch': 5.0,
        'secpr': 800_000.0,
    }
    
    # test gt_mel with mel_lens
    tblogger = deepcopy(tblogger_base)
    batch_dynamic_plots = {
        'gt_mel': torch.randn(1, 640, 160),
        'mel_lens': torch.tensor([640]),
        'audiopath': ['1.wav'],
        'spkrname': ['Cookie'],
    }
    tblogger.plot_media(
        prepend='train', batch=batch_dynamic_plots, batch_dynamic_plots=batch_dynamic_plots, batch_static_plots=None, **common_kwargs
    )
    assert tblogger.summarywriter.plotted_image_tags['train_0/gt_mel'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/gt_mel'].shape) == 3 # image has 3 dims (HWC)
    
    
    
    # test pr_mel with mel_lens
    tblogger = deepcopy(tblogger_base)
    batch_dynamic_plots = {
        'pr_mel': torch.randn(1, 640, 160),
        'mel_lens': torch.tensor([640]),
        'audiopath': ['1.wav'],
        'spkrname': ['Cookie'],
    }
    tblogger.plot_media(
        prepend='train', batch=batch_dynamic_plots, batch_dynamic_plots=batch_dynamic_plots, batch_static_plots=None, **common_kwargs
    )
    assert 'train_0/gt_mel' not in tblogger.summarywriter.plotted_image_tags # make sure gt_mel wasn't plotted if gt_mel wasn't given/wanted
    assert tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape) == 3 # image has 3 dims (HWC)
    
    
    
    # test pr_mel with pr_mel_lens
    tblogger = deepcopy(tblogger_base)
    batch_dynamic_plots = {
        'pr_mel': torch.randn(1, 640, 160),
        'mel_lens': torch.tensor([540]),
        'pr_mel_lens': torch.tensor([640]),
        'audiopath': ['1.wav'],
        'spkrname': ['Cookie'],
    }
    tblogger.plot_media(
        prepend='train', batch=batch_dynamic_plots, batch_dynamic_plots=batch_dynamic_plots, batch_static_plots=None, **common_kwargs
    )
    assert 'train_0/gt_mel' not in tblogger.summarywriter.plotted_image_tags # make sure gt_mel wasn't plotted if gt_mel wasn't given/wanted
    assert tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape) == 3 # image has 3 dims (HWC)
    
    
    
    # test gt_wav with wav_lens
    tblogger = deepcopy(tblogger_base)
    batch_dynamic_plots = {
        'gt_wav': torch.randn(1, 640, 160).clamp(min=-1.0, max=1.0),
        'wav_lens': torch.tensor([640]),
        'audiopath': ['1.wav'],
        'spkrname': ['Cookie'],
    }
    tblogger.plot_media(
        prepend='train', batch=batch_dynamic_plots, batch_dynamic_plots=batch_dynamic_plots, batch_static_plots=None, **common_kwargs
    )
    assert len(tblogger.summarywriter.plotted_audio_tags['train_0/gt_wav'].shape) == 2 # assert audio has 2 dims [BT]
    assert tblogger.summarywriter.plotted_image_tags['train_0/gt_wav'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/gt_wav'].shape) == 3 # image has 3 dims (HWC)
    
    
    # TODO: test gt_wav with pr_wav_lens
    
    # TODO: test pr_wav with wav_lens
    
    # TODO: test pr_wav with pr_wav_lens
    
    # TODO: test pr_mel with mel_lens(max_len=mel_lens.max()+256) # test with pr_mel longer than any lens due to UNet padding
    
    
    
    
    
    # stress test with multiple features and lens with batch_size 1
    tblogger = deepcopy(tblogger_base)
    gt_wav_rec = torch.randn(1, round(math.ceil((640*512)/44100)*44100), 1).clamp(min=-1.0, max=1.0)
    gt_wav_rec[:, 640*512:].fill_(0.0)
    pr_wav = torch.randn(1, round(math.ceil((640*512)/44100)*44100), 1).clamp(min=-1.0, max=1.0)
    pr_wav[:, 640*512:].fill_(0.0)
    pr_mel = torch.randn(1, 1024, 160)
    pr_mel[:, 640:].fill_(0.0)
    batch_dynamic_plots = {
        'gt_wav': torch.randn(1, 640*512, 1).clamp(min=-1.0, max=1.0),
        'gt_wav_rec': gt_wav_rec,
        'pr_wav': pr_wav,
        'gt_mel': torch.randn(1, 640, 160),
        'pr_mel': pr_mel,
        'gt_sf0': torch.randn(1, 640, 1)+5.0,
        'gt_svo': torch.randn(1, 640, 1).sigmoid(),
        'pr_sf0': torch.randn(1, 640, 1)+5.0,
        'pr_svo': torch.randn(1, 640, 1).sigmoid(),
        'mel_lens': torch.tensor([640]),
        'wav_lens': torch.tensor([640*512]),
        'txt_lens': torch.tensor([640//5]),
        'audiopath': ['1.wav'],
        'spkrname': ['Cookie'],
    }
    tblogger.plot_media(
        prepend='train', batch=batch_dynamic_plots, batch_dynamic_plots=batch_dynamic_plots, batch_static_plots=None, **common_kwargs
    )
    assert     tblogger.summarywriter.plotted_image_tags['train_0/gt_mel'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/gt_mel'].shape) == 3 # image has 3 dims (HWC)
    
    assert     tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/pr_mel'].shape) == 3 # image has 3 dims (HWC)
    
    assert     tblogger.summarywriter.plotted_image_tags['train_0/gt_wav_rec'].shape[2] == 3 # assert image is RGB
    assert len(tblogger.summarywriter.plotted_image_tags['train_0/gt_wav_rec'].shape) == 3 # image has 3 dims (HWC)
    
    assert     tblogger.summarywriter.plotted_audio_tags['train_0/gt_wav'].shape[1] == 327680
    assert len(tblogger.summarywriter.plotted_audio_tags['train_0/gt_wav'].shape) == 2 # audio has 2 dims (BT)
    
    assert     tblogger.summarywriter.plotted_audio_tags['train_0/gt_wav_rec'].shape[1] == 327680
    assert len(tblogger.summarywriter.plotted_audio_tags['train_0/gt_wav_rec'].shape) == 2 # audio has 2 dims (BT)
    
    assert     tblogger.summarywriter.plotted_audio_tags['train_0/pr_wav'].shape[1] == 327680
    assert len(tblogger.summarywriter.plotted_audio_tags['train_0/pr_wav'].shape) == 2 # audio has 2 dims (BT)
    
    # TODO: stress test with multiple features and lens with batch_size greater than 1
    
    
    
    w.print1('TensorBoardLogger Test Completed!')

class StandardLogger:
    def __init__(self, standardlogger_config: dict):
        self.show_iteration = standardlogger_config['show_iteration']
        self.show_secpr = standardlogger_config['show_secpr']
        self.show_mean_losses = standardlogger_config['show_mean_losses']
        self.show_grad_norm = standardlogger_config['show_grad_norm']
        self.show_iter_time = standardlogger_config['show_iter_time']
        self.show_dataloading_time = standardlogger_config['show_dataloading_time']
        self.show_learning_rate = standardlogger_config['show_learning_rate']
        self.show_loss_scale = standardlogger_config['show_loss_scale']
    
    def plot_scalars(self, prepend, additional_scalars, loss_dict_reduced,
                grad_norm_total, grad_norm_g, grad_norm_d,
                log_loss_scale, learning_rate, batch_size,
                current_time, iters, epoch, secpr,
                model_time_elapsed, model_secpr_per_second,
                io_time_elapsed, io_secpr_per_second,
                full_time_elapsed, full_secpr_per_second,
                expavg_loss_dict, epochavg_loss_dict,
                bestexpavg_loss_dict, bestepochavg_loss_dict):
        prntlist = [f'{prepend:9}', ]
        if self.show_iteration:
            prntlist.append(f'{iters:>7}')
        if self.show_secpr:
            prntlist.append(f'{secpr:>15,.0f} secpr')
        if self.show_mean_losses:
            if loss_dict_reduced.get("loss_g_reduced", None) is not None:
                prntlist.append(f'{loss_dict_reduced["loss_g_reduced"]:.2f} gloss')
            if loss_dict_reduced.get("loss_d_reduced", None) is not None:
                prntlist.append(f'{loss_dict_reduced["loss_d_reduced"]:.2f} dloss')
        if self.show_grad_norm:
            if grad_norm_g is not None:
                prntlist.append(f'{grad_norm_g:>5.1f} ggrad')
            if grad_norm_d is not None:
                prntlist.append(f'{grad_norm_d:>5.1f} dgrad')
        if self.show_iter_time:
            prntlist.append(f'{full_secpr_per_second:.0f}secpr/s')
            prntlist.append(f'{full_time_elapsed:.1f}s')
        if self.show_dataloading_time:
            prntlist.append(f'{io_time_elapsed:.2f}s io')
        if self.show_learning_rate and learning_rate is not None:
            prntlist.append(f'{learning_rate:.1e} lr'.replace('-0', '-').replace('+0', '+'))
        if self.show_loss_scale and log_loss_scale is not None:
            prntlist.append(f'{log_loss_scale:.0f}lls')
        w.print1("|".join(prntlist))
    
    def plot_media(self, *args, **kwargs):
        pass


def indexed_select(t, select_list): # Tensor, [bool, bool, ...]
    """Select/Index a (list or tensor) using a List of True/False"""
    if type(t) is torch.Tensor:
        if isinstance(select_list, (list, tuple)):
            select_list = torch.tensor(select_list)
        t = t[select_list]
        assert len(t) == sum(select_list), f'got len{len(t)}, expected {sum(select_list)}'
        return t
    elif isinstance(t, (list, tuple)):
        to = []
        for ti, select in zip_equal(t, select_list):
            if select:
                to.append(ti)
        assert len(to) == sum(select_list), f'got len{len(to)}, expected {sum(select_list)}'
        return to
    else:
        return t

class MetricModule:
    """
    Used during training and evaluation.  \n
    1) Takes batch of metrics and predicted features  \n
    2) Process metrics into best values, epoch averages and exp averages  \n
    3) Log using submodules (e.g: Terminal Logger, TextFile Logger, Tensorboard Logger)  \n
    """
    def __init__(self, metricmodule_config: dict, h: dict, run_path: str, rank: int, n_rank: int):
        self.rank = rank
        self.n_rank = n_rank
        
        self.minutes_between_dynamic_plots = metricmodule_config['minutes_between_dynamic_plots']
        self.last_plot_times = {}
        
        # init metric tracking
        self.smoothing_factor = 0.995 # smoothing factor per file
        self.expavg_loss_dict = {} # {{prepend}__{loss_term}: loss_value, ...}
        
        self.epochavg_dict = {}
        # { prepend: {loss_term: MovingDictAverage(audiopath=value)} }
        
        # track best values
        self.lower_better = metricmodule_config['lower_better']
        self.higher_better = metricmodule_config['higher_better']
        self.bestexpavg_loss_dict = {} # {{prepend}__{loss_term}: loss_value, ...}
        self.bestepochavg_loss_dict = {} # {{prepend}__{loss_term}: loss_value, ...}
        
        # media plot params
        self.static_keys: Dict[str, set] = {} # {{prepend}: [key1, key2, ...], ...}
        self.n_static_plots = metricmodule_config['n_static_plots']
        self.n_dynamic_plots = metricmodule_config['n_dynamic_plots']
        
        # init vocoder model
        self.vocoder_config = metricmodule_config['vocoder_config']
        self.vocoder = None
        
        # init various loggers
        if rank == 0:
            self.textfilelogger = None
            self.use_textfilelogger = metricmodule_config['use_textfilelogger']
            
            self.audiofilelogger = None
            self.use_audiofilelogger = metricmodule_config['use_audiofilelogger']
            
            self.tensorboardlogger: TensorBoardLogger
            self.use_tensorboardlogger = metricmodule_config['use_tensorboardlogger']
            if self.use_tensorboardlogger:
                self.tensorboardlogger = TensorBoardLogger(run_path, metricmodule_config['tensorboardlogger_config'], h)
            
            self.standardlogger: StandardLogger
            self.use_standardlogger = metricmodule_config['use_standardlogger']
            if self.use_standardlogger:
                self.standardlogger = StandardLogger(metricmodule_config['standardlogger_config'])
            
            self.wandblogger = None
            self.use_wandblogger = metricmodule_config['use_wandblogger']
    
    def _get_state_keys(self):
        return [
            'last_plot_times', 'expavg_loss_dict',
            'epochavg_dict', # new
#           'epochavg_loss_dict_items', 'epochavg_loss_dict_sum', 'epochavg_loss_dict', # old
            'bestexpavg_loss_dict', 'bestepochavg_loss_dict', 'static_keys',
        ]
    
    def _get_state(self):
        state = {k: getattr(self, k) for k in self._get_state_keys()}
        return state
    
    def load_state(self, state):
        for k, v in state.items():
            if not hasattr(self, k):
                w.print2(f'metricmodule: loaded {k} (but k is not an __init__() attribute)')
            else:
                w.print0(f'metricmodule: loaded {k}')
            setattr(self, k, v)
        
        # Pickle seems to not be respecting __reduce__, __getstate__, __setstate__ or the custom class
        # So manually initializing the class using loaded dict
        epochavg_dict = self.epochavg_dict
        del self.epochavg_dict
        self.epochavg_dict = {}
        for p in list(epochavg_dict.keys()):
            self.epochavg_dict[p] = {}
            for loss_term in list(epochavg_dict[p].keys()):
                self.epochavg_dict[p][loss_term] = MovingDictAverage({k: v for k, v in epochavg_dict[p][loss_term].items()})
    
    def save(self, path):
        w.print1(f'Saving MetricModule to "{path}"')
        safe_write(self._get_state(), path)
    
    def load(self, path):
        w.print1(f'metricmodule: loading from "{path}"')
        state = torch.load(path)
        if any(k not in self._get_state_keys() for k in state):
            w.print2(f'metricmodule state from "{os.path.split(path)[-1]}" is missing keys: {[k for k in state if k not in self._get_state_keys()]}')
        self.load_state(state)
    
    def maybe_use_vocoder(self, batch, device='cuda'):
        # cancel if there aren't any valid files
        if not any(True for k, v in batch.items() if 'mel' in k and 'lens' not in k and v.dim() >= 3 and v.shape[1] > 1 and k.replace('mel', 'wav_rec') not in batch):
            return batch
        
        # maybe load vocoder
        if self.vocoder is None:
            self.vocoder = maybe_load_model(**self.vocoder_config)
        
        if self.vocoder is not None:
            # convert any 'mel' into 'wav' for all mels in batch
            self.vocoder.to(device)
            for k, t in batch.items():
                if type(t) is torch.Tensor and t.dim() >= 3 and t.shape[1] > 1 and 'lens' not in k and 'mel' in k and k.replace('mel', 'wav_rec') not in batch:
                    batch[k.replace('mel', 'wav_rec')] = self.vocoder.infer_specific(t, get_matched_lens(t, [ti for ki, ti in batch.items() if 'lens' in ki]))
            self.vocoder.cpu()
        return batch
    
    def log_step(self, batch: dict, model, additional_scalars: dict, prepend: str, plot_scalars: bool, plot_dynamic_items_media: Optional[bool], model_start_time: Optional[float], dataload_start_time: Optional[float], batch_size: Optional[int], epoch_size: int, rank: int):
        """tracks/logs single batch during continous training/eval"""
        # calc main metrics
        grad_norm_total: Optional[float] = additional_scalars.get('grad_norm_total', None)
        grad_norm_g: Optional[float] = additional_scalars.get('grad_norm_g', None)
        grad_norm_d: Optional[float] = additional_scalars.get('grad_norm_d', None)
        loss_scale: Optional[float] = additional_scalars.get('loss_scale', None)
        learning_rate: Optional[float] = additional_scalars.get('learning_rate', None)
        
        log_loss_scale: Optional[bool] = math.log2(loss_scale) if loss_scale is not None else None
        if batch_size is None:
            batch_size: int = next(t.shape[0] for k, t in batch.items() if type(t) is torch.Tensor and t.dim() >= 3)
            batch_size = dist_add(batch_size, self.n_rank)
        
        current_time = time.time()
        iters: int = model.iteration.item()
        epoch: float = model.epoch.item()
        secpr: float = model.secpr.item()
        secpr_elapsed: float = batch['sec_lens'].sum().item()
        secpr_elapsed = dist_add(secpr_elapsed, self.n_rank)
        
        model_time_elapsed = None
        model_secpr_per_second = None
        if model_start_time is not None:
            model_time_elapsed = current_time - model_start_time
            model_secpr_per_second = secpr_elapsed / model_time_elapsed
        
        io_time_elapsed = None
        io_secpr_per_second = None
        if model_start_time is not None and dataload_start_time is not None:
            io_time_elapsed = model_start_time - dataload_start_time
            io_secpr_per_second = secpr_elapsed / io_time_elapsed
        
        full_time_elapsed = None
        full_secpr_per_second = None
        if dataload_start_time is not None:
            full_time_elapsed = current_time - dataload_start_time
            full_secpr_per_second = secpr_elapsed / full_time_elapsed
        
        # should plot media?
        if prepend not in self.last_plot_times:
            self.last_plot_times[prepend] = 0.0
        if plot_dynamic_items_media is None and current_time > self.last_plot_times[prepend]+(self.minutes_between_dynamic_plots * 60.0):
            self.last_plot_times[prepend] = current_time
            plot_dynamic_items_media = True
        
        # calc main medias
        if plot_dynamic_items_media:
            batch = self.maybe_use_vocoder(batch)
        
        # add reduced collated losses to dict
        loss_dict_reduced = batch['loss_dict'].copy()
        if 'loss_g_reduced' in batch and batch['loss_g_reduced'] is not None:
            loss_dict_reduced['loss_g_reduced'] = batch['loss_g_reduced']
        if 'loss_d_reduced' in batch and batch['loss_d_reduced'] is not None:
            loss_dict_reduced['loss_d_reduced'] = batch['loss_d_reduced']
        if ('loss_g_reduced' in batch and batch['loss_g_reduced'] is not None or
            'loss_d_reduced' in batch and batch['loss_d_reduced'] is not None):
            loss_dict_reduced['loss_total_reduced'] = batch.get('loss_g_reduced', 0.0) + batch.get('loss_d_reduced', 0.0)
        
        # calc exponential average losses
        smoothing_factor = self.smoothing_factor ** batch_size
        for k, v in loss_dict_reduced.items():
            pk = f'{prepend}__{k}'
            if not math.isfinite(v):
                continue
            if pk in self.expavg_loss_dict:
                self.expavg_loss_dict[pk] = (self.expavg_loss_dict[pk]*smoothing_factor) + (v*(1.0-smoothing_factor))
            else:
                self.expavg_loss_dict[pk] = v
        
        try:# calc epoch-size window average loss dict
            if prepend not in self.epochavg_dict:
                self.epochavg_dict[prepend] = {}
            for audiopath, loss_dict in batch['loss_dict_path'].items(): # {audiopath: {loss_term: loss_value}}
                for loss_term, loss_value in loss_dict.items():
                    if not math.isfinite(loss_value):
                        continue
                    if loss_term not in self.epochavg_dict[prepend]:
                        self.epochavg_dict[prepend][loss_term] = MovingDictAverage()
                    self.epochavg_dict[prepend][loss_term][audiopath] = loss_value
                    assert isinstance(self.epochavg_dict[prepend][loss_term], MovingDictAverage), \
                        f'{prepend} : {loss_term} : {audiopath} : {loss_value} : {type(self.epochavg_dict[prepend][loss_term])}'
            
            # calc best values (expavg)
            for k, v in loss_dict_reduced.items():
                pk = f'{prepend}__{k}'
                if pk not in self.bestexpavg_loss_dict:
                    self.bestexpavg_loss_dict[pk] = self.expavg_loss_dict[pk]
                
                if any(hbk in k for hbk in self.higher_better):
                    self.bestexpavg_loss_dict  [pk] = max(self.bestexpavg_loss_dict  [pk], self.expavg_loss_dict[pk])
                else:
                    self.bestexpavg_loss_dict  [pk] = min(self.bestexpavg_loss_dict  [pk], self.expavg_loss_dict[pk])
            # calc best values (epochavg)
            for k, v in self.epochavg_dict[prepend].items():
                pk = f'{prepend}__{k}'
                current_epochavg = self.epochavg_dict[prepend][k].mean()
                
                if pk not in self.bestepochavg_loss_dict:
                    self.bestepochavg_loss_dict[pk] = current_epochavg#self.epochavg_loss_dict[pk]
                
                if any(hbk in k for hbk in self.higher_better):
                    self.bestepochavg_loss_dict[pk] = max(self.bestepochavg_loss_dict[pk], current_epochavg)
                else:
                    self.bestepochavg_loss_dict[pk] = min(self.bestepochavg_loss_dict[pk], current_epochavg)
        except AttributeError:
            print(self.epochavg_dict.keys())
            print(self.epochavg_dict[prepend].keys())
            #print(self.epochavg_dict[prepend][k])
            #print(self.epochavg_dict)
        
        # run loggers
        if rank == 0:
            if prepend not in self.static_keys:
                self.static_keys[prepend] = set()
            if len(self.static_keys[prepend]) < self.n_static_plots:
                for audiopath in batch['audiopath'][:self.n_static_plots-len(self.static_keys[prepend])]:
                    w.print0(f'added {audiopath} to {prepend} static keys')
                    self.static_keys[prepend].add(audiopath)
            
            for audiopath in batch['audiopath']:
                if audiopath in self.static_keys[prepend] and audiopath not in self.last_plot_times:
                    self.last_plot_times[audiopath] = 0.0
            
            select_list = [bool(
                audiopath in self.static_keys[prepend] and
                current_time > self.last_plot_times[audiopath]+(self.minutes_between_dynamic_plots*60.0)) for audiopath in batch['audiopath']]
            
            for audiopath in batch['audiopath']:
                if audiopath in self.static_keys[prepend] and \
                   current_time > self.last_plot_times[audiopath]+(self.minutes_between_dynamic_plots*60.0):
                    self.last_plot_times[audiopath] = current_time
            
            batch_static_plots: Optional[Dict[str, Tensor]]
            if sum(select_list):
                batch_static_plots = Collate().separate_batch(batch, ignore_non_collatable=True) # seperate tensors in each batch
                batch_static_plots = {k: indexed_select(batch_static_plots[k], select_list) for k in batch_static_plots.keys()}
                batch_static_plots = {k: Collate().collate_left(v, k, same_n_channels=not 'alignments' in k) for k, v in batch_static_plots.items()}
            else:
                batch_static_plots = None
            
            if plot_dynamic_items_media:
                batch_dynamic_plots = Collate().separate_batch(batch, ignore_non_collatable=True) # seperate tensors in each batch
                batch_dynamic_plots = {k: batch_dynamic_plots[k][:self.n_dynamic_plots] for k in batch_dynamic_plots.keys()}
                batch_dynamic_plots = {k: Collate().collate_left(v, k, same_n_channels=not 'alignments' in k) for k, v in batch_dynamic_plots.items()}
            else:
                batch_dynamic_plots = None
            
            prepend_startswith = f'{prepend}__'
            pexpavg_loss_dict = {k: v for k, v in self.expavg_loss_dict.items() if k.startswith(prepend_startswith)}
            pepochavg_loss_dict = {f'{prepend}__{k}': v.mean() for k, v in self.epochavg_dict[prepend].items()}
            pbestexpavg_loss_dict = {k: v for k, v in self.bestexpavg_loss_dict.items() if k.startswith(prepend_startswith)}
            pbestepochavg_loss_dict = {k: v for k, v in self.bestepochavg_loss_dict.items() if k.startswith(prepend_startswith)}
            for logger in [self.textfilelogger, self.audiofilelogger, self.tensorboardlogger, self.standardlogger, self.wandblogger]:
                # skip any not-loaded loggers
                if logger is None:
                    continue
                
                # maybe plot scalars
                if plot_scalars and hasattr(logger, 'plot_scalars'):
                    logger.plot_scalars(
                        prepend, additional_scalars, loss_dict_reduced,
                        grad_norm_total, grad_norm_g, grad_norm_d,
                        log_loss_scale, learning_rate, batch_size,
                        current_time, iters, epoch, secpr,
                        model_time_elapsed, model_secpr_per_second,
                        io_time_elapsed, io_secpr_per_second,
                        full_time_elapsed, full_secpr_per_second,
                        pexpavg_loss_dict, pepochavg_loss_dict,
                        pbestexpavg_loss_dict, pbestepochavg_loss_dict
                    )
                
                # maybe plot media
                if hasattr(logger, 'plot_media'):
                    logger.plot_media(
                        prepend, batch, batch_dynamic_plots, batch_static_plots, batch_size, current_time, iters, epoch, secpr,
                    )

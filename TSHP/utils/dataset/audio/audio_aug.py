# imports
import torch
import numpy as np
import random
import librosa
from scipy.signal import butter, sosfilt
from itertools import zip_longest
from math import log2


# testing custom exception handling
import logging
from rich.logging import RichHandler
logging.basicConfig(
    level="WARNING",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")


def zip_equal(*iterables):# taken from https://stackoverflow.com/a/32954700
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError('Iterables have different lengths')
        yield combo

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

class AudioBandpass:
    def __init__(self, enable, min_freq, max_freq, order, sr):
        self.enable = enable
        if self.enable:
            self.min_freq = min_freq
            self.max_freq = max_freq
            self.order    = order
            self.sr       = sr
            self.sos = butter_bandpass(min_freq, max_freq, sr, order)
    
    def __call__(self, audio):
        if not self.enable: return audio
        audio_dtype = audio.dtype
        audio = audio.numpy()
        audio = sosfilt(self.sos, audio.astype('float64'))
        audio = torch.from_numpy(audio).clamp(min=-1.0, max=1.0).to(audio_dtype)
        return audio

class AudioTrim:
    def __init__(self, enable, margin_left, margin_right, ref, top_db, window_length, hop_length, emphasis_str):
        self.enable = enable
        if self.enable:
            self.margin_left   = margin_left
            self.margin_right  = margin_right
            self.ref           = ref
            self.top_db        = top_db
            self.window_length = window_length
            self.hop_length    = hop_length
            self.emphasis_str  = emphasis_str
    
    def __call__(self, audio):
        if not self.enable: return audio
        audio = audio.numpy()
        
        # apply audio trimming
        for i, (
                margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_, preemphasis_strength_
            ) in enumerate(
                zip_equal(self.margin_left, self.margin_right, self.top_db, self.window_length, self.hop_length, self.ref, self.emphasis_str)
            ):
            
            if type(ref_) is str: ref_ = getattr(np, ref_, np.amax)
            preaudio = librosa.effects.preemphasis(audio, coef=preemphasis_strength_) if preemphasis_strength_ else audio
            _, index = librosa.effects.trim(preaudio, top_db=top_db_, frame_length=window_length_, hop_length=hop_length_, ref=ref_) # gonna be a little messed up for different sampling rates
            
            try:
                audio = audio[int(max(index[0]-margin_left_, 0)):int(index[1]+margin_right_)]
            except TypeError:
                print(f'Slice Left:\n{max(index[0]-margin_left_, 0)}\nSlice Right:\n{index[1]+margin_right_}')
            assert len(audio), f"Audio trimmed to 0 length by pass {i+1}\nconfig = {[margin_left_, margin_right_, top_db_, window_length_, hop_length_, ref_]}"
        
        return torch.from_numpy(audio).clamp(min=-1.0, max=1.0).float()

class AudioLUFSNorm:
    def __init__(self, sr, target_lufs):
        self.sr = sr
        self.target_lufs = target_lufs
        if self.target_lufs is not None:
            import pyloudnorm as pyln
            self.meter = pyln.Meter(sr)# create BS.1770 meter
    
    def get_lufs_vol(self, audio):
        if not hasattr(self, 'meter'):
            import pyloudnorm as pyln
            self.meter = pyln.Meter(self.sr)# create BS.1770 meter
        lufs = self.meter.integrated_loudness(audio.squeeze().numpy())# measure loudness (in dB)
        return torch.tensor(lufs).float()
    
    def __call__(self, audio, audiopath=''):
        if self.target_lufs is None: return audio
        try:
            lufs = self.get_lufs_vol(audio)
            
            if type(lufs) is torch.Tensor:
                lufs = lufs.to(audio)
            delta_lufs = self.target_lufs-lufs
            assert torch.isfinite(lufs).all(), f'Original LUFS is non-finite. Got {lufs}. Original Audio std = {audio.std().item()}'
            gain = 10.0**(delta_lufs/20.0)
            assert torch.isfinite(gain).all(), f'gain is non-finite. Got {gain}'
            audio = audio*gain
            assert torch.isfinite(audio).all(), 'returned audio output for volume norm has non-finite elements.'
            if audio.abs().max() > 1.0:
                numel_over_limit = (audio.abs() > 1.0).sum()
                if numel_over_limit > audio.numel()/(self.sr/16):# if more than 16 samples per second are over 1.0, do peak normalization. Else just clamp them.
                    if False:#self.warnings:# disabled warning. I haven't ran into a circumstance where it's helpful, it seems to print on normal files just as much as broken ones.
                        print(f'Found {numel_over_limit} mags over 1.0. Max mag was {audio.abs().max().item()}. Normalizing peak to 1.0.')
                    audio /= audio.abs().max()
                audio.clamp_(min=-1.0, max=1.0)
        except AssertionError:
            log.exception(f"Unable to normalize audio.\npath = '{audiopath}'")
        except ValueError:
            log.exception(f"Unable to normalize audio.\npath = '{audiopath}'")
        return audio


# audio transforms
class AudioAug:
    def __init__(self,
            rescale_pitch,  rescale_pitch_prob,  max_pitch_rescale,  min_pitch_rescale,
            rescale_speed,  rescale_speed_prob,  max_speed_rescale,  min_speed_rescale,
            rescale_volume, rescale_volume_prob, max_volume_rescale, min_volume_rescale,
            add_whitenoise, add_whitenoise_prob, max_whitenoise_db,  min_whitenoise_db,
            add_noise,      add_noise_prob,      max_noise_db,       min_noise_db, noise_dataset,
            sr, seed = 1234,
        ):
        (self.b_rescale_pitch,  self.rescale_pitch_prob,  self.max_pitch_rescale,  self.min_pitch_rescale,
         self.b_rescale_speed,  self.rescale_speed_prob,  self.max_speed_rescale,  self.min_speed_rescale,
         self.b_rescale_volume, self.rescale_volume_prob, self.max_volume_rescale, self.min_volume_rescale,
         self.b_add_whitenoise, self.add_whitenoise_prob, self.max_whitenoise_db,  self.min_whitenoise_db,
         self.b_add_noise,      self.add_noise_prob,      self.max_noise_db,       self.min_noise_db, self.noise_dataset, self.sr) = (
            rescale_pitch,  rescale_pitch_prob,  max_pitch_rescale,  min_pitch_rescale,
            rescale_speed,  rescale_speed_prob,  max_speed_rescale,  min_speed_rescale,
            rescale_volume, rescale_volume_prob, max_volume_rescale, min_volume_rescale,
            add_whitenoise, add_whitenoise_prob, max_whitenoise_db,  min_whitenoise_db,
            add_noise,      add_noise_prob,      max_noise_db,       min_noise_db, noise_dataset, sr,
        )
        self.random = random.Random(seed)
    
    def update_speed(self, audio):
        # use resample func to update speed
        speed_change = self.random.uniform(self.min_speed_rescale, self.max_speed_rescale) if self.random.random() < self.rescale_speed_prob else 1.0
        if speed_change != 1.0:
            audio = librosa.effects.time_stretch(audio, speed_change)
        return audio
    
    def update_pitch(self, audio):
        # +1 octave = double f0
        pitch_scalar = self.random.uniform(self.min_pitch_rescale, self.max_pitch_rescale) if self.random.random() < self.rescale_pitch_prob else 1.0
        octave_offset = log2(pitch_scalar)
        if pitch_scalar != 1.0:
            audio = librosa.effects.pitch_shift(audio, self.sr, octave_offset, bins_per_octave=1)
        return audio
    
    def update_volume(self, audio):
        audio_dB_offset = self.random.uniform(self.min_volume_rescale, self.max_volume_rescale) if self.random.random() < self.rescale_volume_prob else 0.0
        if audio_dB_offset != 0.0:
            audio_scalar = 10**(audio_dB_offset/10.0)
            audio = audio*audio_scalar
        return audio
    
    def add_whitenoise(self, audio):
        audio = torch.from_numpy(audio)
        noise = torch.randn(audio.shape, device=audio.device, dtype=audio.dtype)
        # incomplete
        return audio.numpy()
    
    def add_noise(self, audio):
        # incomplete
        return audio
    
    def __call__(self, audio):
        audio = audio.numpy()
        if self.b_rescale_speed:
            audio = self.update_speed(audio)
        if self.b_rescale_pitch:
            audio = self.update_pitch(audio)
        if self.b_rescale_volume:
            audio = self.update_volume(audio)
        if self.b_add_whitenoise:
            audio = self.add_whitenoise(audio)
        if self.b_add_noise:
            audio = self.add_noise(audio)
        return torch.from_numpy(audio)
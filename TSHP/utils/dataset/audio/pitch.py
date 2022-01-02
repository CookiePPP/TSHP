# imports
from typing import Optional
from torch import Tensor

import numpy as np
import torch
import pyworld as pw


def get_pitch(audio, sr, hop_len, f0_min, f0_max, f0_refine, voiced_sensitivity):# FloatTensor[wav_T]
        """ Extract Pitch/f0 from raw waveform using PyWORLD """
        audio = audio.view(-1)
       # audio = torch.cat((audio, audio[-1:]), dim=0)
        audio = audio.numpy().astype(np.float64)
        
        pitch, timeaxis = pw.dio(# get raw logpitch
            audio, sr,
            frame_period=(hop_len/sr)*1000.,# For hop size 256 frame period is 11.6 ms
            f0_floor=f0_min,
            f0_ceil =f0_max,
            allowed_range=voiced_sensitivity,
        )
        if f0_refine:
            pitch = pw.stonemask(audio, pitch, timeaxis, sr)# logpitch refinement
        pitch = torch.from_numpy(pitch).float().clamp(min=0.0, max=f0_max)# (Number of Frames) = (654,)
        voiced = (pitch>1)# voice / unvoiced flag
        
        assert not (torch.isinf(pitch) | torch.isnan(pitch)).any(), f"f0 from pyworld is NaN. Info below\nlen(audio) = {len(audio)}\nf0 = {pitch}\nsampling_rate = {sr}"
        return pitch, voiced# [mel_T], [mel_T]
    

# get logpitch
class PitchModule:
    def __init__(self, sr, hop_len,
            f0_min, f0_max, f0_refine, voiced_sensitivity, soft_config
        ):
        self.sr      = sr
        self.hop_len = hop_len
        self.f0_min    = f0_min
        self.f0_max    = f0_max
        self.f0_refine = f0_refine
        self.voiced_sensitivity = voiced_sensitivity
        self.voiced_sensitivities = soft_config['voiced_sensitivities']
    
    def get_pitch(self, audio):# [wav_T]
        """ Extract Pitch/f0 from raw waveform using PyWORLD """
        pitch, voiced = get_pitch(audio, self.sr, self.hop_len, self.f0_min, self.f0_max, self.f0_refine, self.voiced_sensitivity)
        return pitch, voiced# [mel_T], [mel_T]
    
    def get_softpitch(self, audio):# [wav_T]
        soft_pitch = None
        soft_voiced = []
        for voiced_sensitivity in self.voiced_sensitivities:
            pitch, voiced = get_pitch(audio, self.sr, self.hop_len, self.f0_min, self.f0_max, self.f0_refine, voiced_sensitivity)
            soft_voiced.append(voiced.float())
            if soft_pitch is None:
                soft_pitch = pitch
            soft_pitch = pitch.where(voiced, soft_pitch)
        soft_voiced = torch.stack(soft_voiced, dim=0).mean(dim=0)
        return soft_pitch, soft_voiced# [mel_T], [mel_T]
    
    def get_logpitch_from_pitch(self, pitch, voiced=None):
        if voiced is None:
            voiced = self.get_voiced_from_pitch(pitch)
        return pitch.log().where(voiced, pitch)
    
    def get_voiced_from_pitch(self, pitch):
        return pitch>1
    
    def get_logpitch(self, audio):# [wav_T]
        pitch, voiced = self.get_pitch(audio)
        pitch = self.get_logpitch_from_pitch(pitch, voiced)
        return pitch, voiced# [mel_T], [mel_T]
    
    def get_logsoftpitch(self, audio):# [wav_T]
        soft_pitch, soft_voiced = self.get_softpitch(audio)
        soft_pitch = self.get_logpitch_from_pitch(soft_pitch, soft_voiced)
        return soft_pitch, soft_voiced# [mel_T], [mel_T]
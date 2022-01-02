# imports
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np

def mag_to_log(x, clamp_val=None, logtype=None):
    logtype = logtype
    if not type(x) is torch.Tensor:
        x = torch.tensor(x)
    if clamp_val:
        x = x.clamp(min=clamp_val)
    if   logtype == 'log':   return x.log()
    elif logtype == 'log10': return x.log10()
    elif logtype == 'db':    return x.log10()*10.0
    elif logtype == 'log2':  return x.log2()
    else: raise NotImplementedError(f'logtype of {logtype} is not valid.')

def log_to_mag(x, logtype=None):
    logtype = logtype
    if not type(x) is torch.Tensor: x = torch.tensor(x)
    if   logtype == 'log':   return x.exp()
    elif logtype == 'log10': return 10**x
    elif logtype == 'db':    return 10**(x/10.0)
    else: raise NotImplementedError(f'logtype of {logtype} is not valid.')

# stft
class STFTModule(nn.Module):
    def __init__(self, sr, fil_len, hop_len, win_len, clamp_val, log_type,
            stft_norm, mel_norm, n_mel, fmin, fmax):
        super().__init__()
        self.fil_len   = fil_len
        self.hop_len   = hop_len
        self.win_len   = win_len
        self.clamp_val = clamp_val
        self.log_clamp_val = mag_to_log(self.clamp_val, 0, log_type).item()
        self.log_type  = log_type
        self.stft_norm = stft_norm
        
        self.n_mel    = n_mel
        self.fmin     = fmin
        self.fmax     = min(fmax, sr/2.0) if fmax is not None else sr/2.0
        self.mel_norm = mel_norm
        
        # get window for STFT
        self.window = torch.hann_window(self.win_len)
        
        # get melscale for Mels
        self.melscale = torchaudio.transforms.MelScale(
            n_mel, sr, fmin, fmax, self.fil_len//2+1)
        
        # get frequency weighting for perceieved volume
        self.sr = sr
        freqs = np.clip(librosa.fft_frequencies(sr=sr, n_fft=self.fil_len), 0.1, None)
        self.freq_weights = torch.from_numpy(librosa.frequency_weighting(freqs, 'A'))
    
    def from_audio_get_spect(self, audio):# [..., wav_T, 1]
        assert(torch.min(audio) >= -1.), f'Tensor.min() of {torch.min(audio).item()} is less than -1.0'
        assert(torch.max(audio) <=  1.), f'Tensor.max() of {torch.max(audio).item()} is greater than 1.0'
        
        audio = audio.squeeze(-1)# [..., wav_T]
        
        # trim so wav_T//hop_len == mel_T under all circumstances.
        audio = audio[..., :-1]
        assert audio.shape[-1] >= self.fil_len, f'got audio length of {audio.shape[-1]}, expected {self.fil_len} or more. shape = {audio.shape}'
        spec = torch.stft(audio.float(),
            n_fft     =self.fil_len,
            hop_length=self.hop_len,
            win_length=self.win_len,
            window    =self.window.to(audio).float(),
            return_complex=True,
        ).abs().to(audio)# -> [..., n_stft, mel_T, 2]
        return spec.transpose(-1, -2)# [..., mel_T, n_stft]
    
    def from_audio_get_perc_loudness(self, audio):# [..., wav_T, 1]
        stft = self.from_audio_get_spect(audio)# -> [..., mel_T, n_stft]
        loudness = self.from_spect_get_perc_loudness(stft)
        return loudness
    
    def from_spect_get_mel(self, spec):# [..., mel_T, n_stft]
        device = next(self.melscale.buffers())
        mel = self.mag_to_log(self.melscale(spec.to(device).transpose(-1, -2)).transpose(-1, -2), logtype=self.log_type).to(spec)
        return mel
    
    def from_audio_get_mel(self, audio):# [..., wav_T, 1]
        spec = self.from_audio_get_spect(audio)# [..., mel_T, n_stft]
        mel = self.from_spect_get_mel(spec)# [..., mel_T, n_stft]
        return mel
    
    def from_spect_get_perc_loudness(self, spec):# [..., mel_T, n_stft]
        db_spec = self.mag_to_log(spec, logtype='db')
        freq_weights = self.freq_weights
        for i in range(freq_weights.dim()-1):
            freq_weights = freq_weights.unsqueeze(0)# [..., 1, n_stft]
        spec = self.log_to_mag(freq_weights+db_spec, logtype='db').clamp(min=self.clamp_val)
        loudness = (spec.mean(-1)+ 1e-5).log().unsqueeze(-1)# [..., mel_T, 1]
        return loudness
    
    def mag_to_log(self, x, logtype=None):
        return mag_to_log(x, self.clamp_val, logtype)
    
    def log_to_mag(self, x, logtype=None):
        return log_to_mag(x, logtype)
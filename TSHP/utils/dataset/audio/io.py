import numpy as np
import torch
import soundfile as sf
import librosa
from scipy.io.wavfile import read

def get_audio_from_path(full_path, target_sr=None, min_sr=None, remove_dc_offset=True, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        if return_empty_on_exception:
            print(ex)
            return [], sampling_rate or target_sr or 1
        else:
            raise ex
    
    if min_sr is not None:
        if return_empty_on_exception and not (min_sr < sampling_rate):
            return [], sampling_rate or target_sr or 1
        assert min_sr < sampling_rate, f'Expected sampling_rate greater than or equal to {min_sr:.0f}, got {sampling_rate:.0f}.\nPath = "{full_path}"'
    
    if len(data.shape) > 1: # if audio has more than 1 channels,
        if data.shape[1] > data.shape[0] and data.shape[1] > 6:
            data = data.transpose(0, 1)
        data = data[:, 0]   # extract/use the first channel.
    
    #if np.issubdtype(data.dtype, np.integer): # if audio data is type int
    #    max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    #else: # if audio data is type fp32
    #    max_mag = max(np.amax(data), -np.amin(data))
    #    if max_mag > (2**23):
    #        max_mag = 2**31
    #    elif max_mag > (2**15):
    #        max_mag = 2**23
    #    elif max_mag > (1.99):
    #        max_mag = 2**15
    #    else:
    #        max_mag = 1.0
    #data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    data = torch.FloatTensor(data.astype(np.float32))
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# check for Nan/Inf in audio files
        return [], sampling_rate or target_sr or 1
    assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found in audio file\n"{full_path}"'
    
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), sampling_rate, target_sr))
        if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
            return [], sampling_rate or target_sr or 1
        assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found after resampling audio\n"{full_path}"'
        
        sampling_rate = target_sr
    
    if remove_dc_offset:
        data -= data.mean()
        assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found after removing DC offset\n"{full_path}"'
    
    abs_max = data.abs().max()
    if abs_max > 1.0:
        data /= abs_max
        assert not (torch.isinf(data) | torch.isnan(data)).any(), f'Inf or NaN found after inf-norm rescaling audio\n"{full_path}"'
    
    return data, sampling_rate
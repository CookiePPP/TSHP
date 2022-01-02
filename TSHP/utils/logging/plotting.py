import math
from typing import Optional, List, Tuple

import matplotlib
from matplotlib.pyplot import yticks

matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np

##############################################################################################
# Plotting code taken from NVIDIA/tacotron2 and modified with clim, title and yticks options #
##############################################################################################

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def set_range(im, clim: Tuple[float, float]):
    if clim is not None:
        assert len(clim) == 2, 'range params should be a 2 element List of [Min, Max].'
        assert clim[1] > clim[0], 'Max (element 1) must be greater than Min (element 0).'
        im.set_clim(clim[0], clim[1])


def plot_alignment_to_numpy(
        alignment: np.ndarray, # [txt_T, mel_T]
        text_symbols: Optional[List[str]] = None,
        info: str = None,
        title: str = None,
        clim: Optional[Tuple[float, float]] = None):
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
        yticks(range(len(text_symbols)), text_symbols)
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

import torch
def _hz_to_mel(freq: torch.Tensor, mel_scale: str = "htk") -> torch.Tensor:
    r"""Convert Hz to Mels.
    
    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)
    
    Returns:
        mels (float): Frequency in Mels
    """

    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * (1.0 + (freq / 700.0)).log10()
    
    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3
    
    mels = (freq - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    
    mels = mels.where(freq < min_log_hz, min_log_mel + (freq / min_log_hz).log() / logstep)
    
    return mels

if __name__ == '__main__':
    f0 = torch.tensor([100.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 9600.0, 19200.0])
    mel = _hz_to_mel(f0).div(_hz_to_mel(torch.tensor(18000.0))).mul(160)
    print(mel)

def plot_spectrogram_to_numpy(spectrogram, clim=None, title=None, logpitch=None):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, cmap='gist_ncar', aspect="auto", origin="lower",
                   interpolation='none')
    if logpitch is not None:
        ax.scatter(range(len(logpitch)), _hz_to_mel(logpitch.exp()).sub(_hz_to_mel(torch.tensor(20.0))).clamp(min=0.0).div(_hz_to_mel(torch.tensor(11025.0))).mul(160).numpy(), alpha=1.0,
                   color='blue', marker='+', s=1, label='f0')
    set_range(im, clim)
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_time_series_to_numpy(target,# Numpy/FloatTensor[T]
                                pred,# Numpy/FloatTensor[T]
                          pred2=None,# Numpy/FloatTensor[T]
                           xlabel="Frames (Green target, Red predicted)",
                           ylabel="Gate State",
                            title=None,
                             ymin=None,
                             ymax=None,
                            alpha=0.5,
                          alpha_a=None,
                          alpha_b=None,
                          alpha_c=None,
                          color_a='green',
                          color_b='red',
                          color_c='blue',
                          label_a='target',
                          label_b='predicted',
                          label_c='predicted',):
    assert target is not None or pred is not None or pred2 is not None, 'got None for all input Tensors' 
    assert target is None or len(target.shape) == 1, f'got target.shape of {target.shape}, expected [T] (where T is length of tensor)'
    assert pred is None or len(pred.shape) == 1, f'got target.shape of {pred.shape}, expected [T] (where T is length of tensor)'
    assert pred2 is None or len(pred2.shape) == 1, f'got target.shape of {pred2.shape}, expected [T] (where T is length of tensor)'
    
    fig, ax = plt.subplots(figsize=(12, 3))
    if target is not None:
        ax.scatter(range(len(target)), target, alpha=alpha_a or alpha,
                   color=color_a, marker='+', s=1, label=label_a)
    if pred is not None:
        ax.scatter(range(len(pred)), pred, alpha=alpha_b or alpha,
                   color=color_b, marker='_', s=1, label=label_b)
    if pred2 is not None:
        ax.scatter(range(len(pred2)), pred2, alpha=alpha_c or alpha,
                   color=color_c, marker='_', s=1, label=label_c)
    
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)
    
    if title is not None:
        assert isinstance(title, str), f'title of "{title}" is not type str'
        plt.title(title)
    if xlabel is not None:
        assert isinstance(xlabel, str), f'xlabel of "{xlabel}" is not type str'
        plt.xlabel(xlabel)
    if ylabel is not None:
        assert isinstance(ylabel, str), f'ylabel of "{ylabel}" is not type str'
        plt.ylabel(ylabel)
    plt.tight_layout()
    
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data # npy[H, W, RGB]


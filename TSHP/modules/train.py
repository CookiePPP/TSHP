# imports
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy

from typing import Optional

import yaml
import random

import torch
import torch.distributed as dist

from TSHP.modules.train_optim import OptimizerModule, MovingPercentile, MovingDictAverage
from TSHP.utils.distributed.ModelDataParallel import apply_gradient_allreduce
from TSHP.utils.downloads.download_urls import download_unknown

from TSHP.utils.modules.core import ModelModule, dist_add, dist_mean, dist_barrier, specific_rank_first

from TSHP.utils.misc_utils import deepupdate_dicts, zip_equal

from TSHP.modules.train_logger import MetricModule, get_matched_lens
from TSHP.utils.dataset.dataset import filelist, DataLoaderModule, DatasetModule, Collate
from torch.utils.data import DistributedSampler

from TSHP.utils.saving.utils import safe_write

import logging

from TSHP.utils.load_ctts2_checkpoints import load_model_from_path
from TSHP.modules.train_utils import find_weight_path, deepto

log = logging.getLogger('rich')
import TSHP.utils.warnings as w

# these are needed for fp16 training, not inference
try:
    from apex import amp
except ImportError:
    class amp: # make mock amp if AMP is not installed
        def master_params(*args, **kwargs):
            raise NotImplementedError
        
        def state_dict(*args, **kwargs):
            raise NotImplementedError
        
        def scale_loss(*args, **kwargs):
            raise NotImplementedError
        
        def initialize(*args, **kwargs):
            raise NotImplementedError

if __name__ == '__main__':
    w.setLevel('info')

class PreModelModule:
    """
    Used during training.  \n
    1) Takes batch (from dataloader)  \n
    2) Runs any pre-processing models on batch with GPU  \n
    3) Returns batch
    """
    def __init__(self, premodel_config):
        pass
    
    def __call__(self, pr):
        
        return pr

class PostModelModule:
    """
    Used during training.  \n
    1) Takes batch (of predicted features)  \n
    2) Uses conversion models and metric models together to add additional metrics to batch  \n
    3) Returns batch
    """
    def __init__(self, postmodel_config, n_mel, preload=False):
        self.n_mel = n_mel
        
        self.hcp2mel_path = postmodel_config['hcp2mel_path']
        self.hcp2frp_path = postmodel_config['hcp2frp_path']
        self.frp2mel_path = postmodel_config['frp2mel_path']
        self.textctc_path = postmodel_config['textctc_path']
        self.spkrsim_path = postmodel_config['spkrsim_path']
        self.vocoder_path = postmodel_config['vocoder_path']
        
        self.hcp2mel = None
        self.hcp2frp = None
        self.frp2mel = None
        self.textctc = None
        self.spkrsim = None
        self.vocoder = None
        
        if preload:
            if self.hcp2mel_path:
                self.hcp2mel, _ = load_model_from_path(self.hcp2mel_path)
            if self.hcp2frp_path:
                self.hcp2frp, _ = load_model_from_path(self.hcp2frp_path)
            if self.frp2mel_path:
                self.frp2mel, _ = load_model_from_path(self.frp2mel_path)
            if self.textctc_path:
                self.textctc, _ = load_model_from_path(self.textctc_path)
            if self.spkrsim_path:
                self.spkrsim, _ = load_model_from_path(self.spkrsim_path)
            if self.vocoder_path:
                self.vocoder, _ = load_model_from_path(self.vocoder_path)
                
    def hcp2mel_path(self, pr_dur, pr_hc_nrg, pr_hc_svo, pr_hc_logsf0, txt_lens, mel_lens=None):
        # load hcp2mel
        # move to GPU
        # run model
        return mel, mel_lens
    
    def run_hcp2frp(self, pr_dur, pr_hc_nrg, pr_hc_svo, pr_hc_logsf0, txt_lens, mel_lens=None):
        # load hcp2frp
        # move to GPU
        # run model
        return pr_nrg, pr_svo, pr_logsf0, mel_lens
    
    def run_frp2mel(self, pr):
        if self.frp2mel_path is None:
            return pr
        
        if not 'pr_nrg' in pr and 'pr_frp' in pr:
            pr['pr_nrg'] = pr['pr_frp'][:, :, 0:1]
        
        if not 'pr_svo' in pr and 'pr_frp' in pr:
            pr['pr_svo'] = pr['pr_frp'][:, :, 1:2]
        
        if not 'pr_logsf0' in pr and 'pr_frp' in pr:
            pr['pr_logsf0'] = pr['pr_frp'][:, :, 2:3]
        
        if not ('pr_dur' in pr and 'pr_nrg' in pr and 'pr_svo' in pr and 'pr_logsf0' in pr):
            return pr
        
        # load frp2mel
        if self.frp2mel is None:
            self.frp2mel, _ = load_model_from_path(self.frp2mel_path)
        
        # move to GPU
        self.frp2mel.cuda()
        
        # run model
        pr['text_ids'], pr['txt_lens'], pr['spkr_ids'], pr['pr_dur'], pr['pr_nrg'], pr['pr_svo'], pr['pr_logsf0'] = self.vocoder.transfer_device([
            pr['text_ids'], pr['txt_lens'], pr['spkr_ids'], pr['pr_dur'], pr['pr_nrg'], pr['pr_svo'], pr['pr_logsf0']])
        d = self.frp2mel.generator.infer(
            pr['text_ids'], pr['txt_lens'], pr['spkr_ids'], pr['pr_dur'], pr['pr_nrg'], pr['pr_svo'], pr['pr_logsf0'])
        pr['mel_lens'] = d['mel_lens']
        pr['pr_mel'] = d['pr_mel'] # [B, mel_T, n_mel]
        pr['alignments'] = d['alignments']
        
        return pr
    
    def run_textctc(self, mel, mel_lens, text_ids, txt_lens):
        # load textctc
        # move to GPU
        # run model
        return text_ctc
    
    def run_spkrsim(self, mel1, mel1_lens, mel2, mel2_lens):
        # load spkrsim
        # move to GPU
        # run model
        return spkr_cossim, spkr_cosdist
    
    def run_vocoder(self, pr):
        if self.vocoder_path is None:
            return pr
        # load vocoder
        if self.vocoder is None:
            self.vocoder, _ = load_model_from_path(self.vocoder_path)
        
        # move to GPU
        self.vocoder.cuda()
        
        # run model
        for k in list(pr.keys()):
            if type(pr[k]) is not torch.Tensor:
                continue
            if 'lens' in k:
                continue
            if pr[k].shape[2] != self.n_mel:
                continue
            if 'mel' in k and k.replace('mel', 'wav_rec') not in pr:
                mel, mel_lens = pr[k], get_matched_lens(pr[k], [v for k, v in pr.items() if 'lens' in k])
                mel, mel_lens = self.vocoder.transfer_device([mel, mel_lens])
                d = self.vocoder.generator.infer(None, mel, mel_lens)
                pr[k.replace('mel', 'wav_rec')], pr['wav_lens_rec'] = d['pr_wav'], d['wav_lens']
        
        return pr
    
    def __call__(self, pr):
        with torch.no_grad():
            
            #mel, mel_lens = self.hcp2mel_path(pr_dur, pr_hc_nrg, pr_hc_svo, pr_hc_logsf0, txt_lens, mel_lens=None)
            
            #pr_nrg, pr_svo, pr_logsf0, mel_lens = self.run_hcp2frp(pr_dur, pr_hc_nrg, pr_hc_svo, pr_hc_logsf0, txt_lens, mel_lens=None)
            
            pr = self.run_frp2mel(pr)
            
            #text_ctc = self.run_textctc(mel, mel_lens, text_ids, txt_lens)
            
            #spkr_cossim, spkr_cosdist = self.run_spkrsim(mel1, mel1_lens, mel2, mel2_lens)
            
            pr = self.run_vocoder(pr)
        return pr


if __name__ == '__main__':
    # test PostModelModule
    
    # TODO: test init PostModelModule with empty paths
    postmodel_config = {
        'hcp2mel_path': '',
        'hcp2frp_path': '',
        'frp2mel_path': '',
        'textctc_path': '',
        'spkrsim_path': '',
        'vocoder_path': '',
    }
    pmm = PostModelModule(postmodel_config, preload=True, n_mel=160)
    assert pmm.hcp2mel is None
    assert pmm.hcp2frp is None
    assert pmm.frp2mel is None
    assert pmm.textctc is None
    assert pmm.spkrsim is None
    assert pmm.vocoder is None
    
    
    # TODO: test init PostModelModule with None paths
    postmodel_config = {
        'hcp2mel_path': None,
        'hcp2frp_path': None,
        'frp2mel_path': None,
        'textctc_path': None,
        'spkrsim_path': None,
        'vocoder_path': None,
    }
    pmm = PostModelModule(postmodel_config, preload=True, n_mel=160)
    assert pmm.hcp2mel is None
    assert pmm.hcp2frp is None
    assert pmm.frp2mel is None
    assert pmm.textctc is None
    assert pmm.spkrsim is None
    assert pmm.vocoder is None
    
    # TODO: test init PostModelModule with all real paths
    postmodel_config = {
        'hcp2mel_path': '',
        'hcp2frp_path': '',
        'frp2mel_path': "I:\\csruns\\frp_to_mel\\DPMMELD\\outdir_11_Clipper\\weights\\best_cross_val.ptw",
        'textctc_path': '',
        'spkrsim_path': '',
        'vocoder_path': "I:\\csruns\\vocoder\\FreGAN\\outdir_11_Pandora\\weights\\best_cross_val.ptw",
    }
    pmm = PostModelModule(postmodel_config, preload=True, n_mel=160)
    assert pmm.hcp2mel is not None or not postmodel_config['hcp2mel_path']
    assert pmm.hcp2frp is not None or not postmodel_config['hcp2frp_path']
    assert pmm.frp2mel is not None or not postmodel_config['frp2mel_path']
    assert pmm.textctc is not None or not postmodel_config['textctc_path']
    assert pmm.spkrsim is not None or not postmodel_config['spkrsim_path']
    assert pmm.vocoder is not None or not postmodel_config['vocoder_path']
    
    # TODO: test PostModelModule with gt_mel + mel_lens (test vocoder)
    gt_mel = torch.randn(2, 640, 160)
    mel_lens = torch.tensor([640])
    pr = {
        'gt_mel': gt_mel,
        'mel_lens': mel_lens,
    }
    pr = pmm(pr)
    assert pmm.vocoder is not None
    assert 'gt_wav_rec' in pr
    assert pr['gt_wav_rec'].shape[1] == pr['gt_mel'].shape[1]*pmm.vocoder.generator.hop_len
    assert 'wav_lens_rec' in pr
    
    # TODO: test PostModelModule with pr_mel + mel_lens (test vocoder)
    pr_mel = torch.randn(2, 640, 160)
    mel_lens = torch.tensor([640])
    pr = {
        'pr_mel': pr_mel,
        'mel_lens': mel_lens,
    }
    pr = pmm(pr)
    print({k: v.shape for k, v in pr.items() if type(v) is torch.Tensor})
    assert pmm.vocoder is not None
    assert 'pr_wav_rec' in pr
    assert 'wav_lens_rec' in pr
    
    # TODO: test PostModelModule with pr_mel_z + mel_lens (test vocoder)
    # (in this case, it's a tensor with 'mel' but it's not a spectrogram and should be skipped
    pr_mel_z = torch.randn(2, 640, 72)
    mel_lens = torch.tensor([640])
    pr = {
        'pr_mel_z': pr_mel_z,
        'mel_lens': mel_lens,
    }
    pr = pmm(pr)
    assert pmm.vocoder is not None
    assert 'pr_wav_rec' not in pr
    
    # TODO: test PostModelModule with pr_frp + mel_lens + pr_dur + txt_lens (test frp_to_mel)
    pr_frp = torch.randn(2, 640, 3) # [gt_nrg, gt_svo, gt_logsf0]
    pr_dur = torch.rand(2, 144, 1).mul(2).long()
    pr_dur[:, -1] = 640 - pr_dur[:, :-1].sum(1)
    mel_lens = torch.tensor([640, 640])
    text_ids = torch.rand(2, 144, 1).mul(100).long()
    spkr_ids = torch.rand(2, 1, 1).mul(10).long()
    txt_lens = torch.tensor([144, 144])
    pr = {
        'pr_dur': pr_dur,
        'pr_frp': pr_frp,
        'mel_lens': mel_lens,
        'text_ids': text_ids,
        'txt_lens': txt_lens,
        'spkr_ids': spkr_ids,
    }
    pr = pmm(pr)
    assert pmm.vocoder is not None
    assert 'pr_wav_rec' in pr
    
    # TODO: test PostModelModule with pr_frp + mel_lens + hard_alignments (test frp_to_mel)
    #pr_frp = torch.randn(2, 640, 3) # [gt_nrg, gt_svo, gt_logsf0]
    #mel_lens = torch.tensor([640])
    #txt_lens = torch.tensor([144])
    #alignments = torch.randn(2, 144, 640).softmax(dim=1)
    #hard_alignments = Viterbi(alignments)
    #pr = {
    #    'pr_frp': pr_frp,
    #    'mel_lens': mel_lens,
    #    'txt_lens': txt_lens,
    #    'hard_alignments': hard_alignments,
    #}
    #pr = pmm(pr)
    #assert pmm.vocoder is not None
    #assert 'pr_wav_rec' in pr
    
    
    # TODO: test PostModelModule with pr_frp + mel_lens + alignments + txt_lens (test frp_to_mel)
    #pr_frp = torch.randn(2, 640, 3) # [gt_nrg, gt_svo, gt_logsf0]
    #mel_lens = torch.tensor([640])
    #txt_lens = torch.tensor([144])
    #pr_dur = torch.randn(2, 144, 1).add(1.0).exp()
    #pr = {
    #    'pr_frp': pr_frp,
    #    'mel_lens': mel_lens,
    #    'txt_lens': txt_lens,
    #    'pr_dur': pr_dur,
    #}
    #pr = pmm(pr)
    #assert pmm.vocoder is not None
    #assert 'pr_frp_to_wav_rec' in pr
    
    # TODO: test PostModelModule with pr_frpn + mel_lens (test frp_to_mel)
    
    
    # TODO: test PostModelModule with pr_hcp + mel_lens (test hcp_to_frp)
    
    
    # TODO: test PostModelModule with pr_hcpn + mel_lens (test hcp_to_frp)
    
    
    # TODO: test speaker similarity
    
    
    # TODO: test ambiguous lengths
    
    
    # TODO: sliced input feats
    
    
    
    w.print1('PostModelModule Test Completed!')

class TrainModelModule:
    """
    Used during training.  \n
    1) Takes batch  \n
    2) runs model + evaluation models.  \n
    3) Returns batch features, loss terms and eval metrics
    """
    def __init__(self, modelmodule_config, weight_path, model_identity, h):
        self.model: ModelModule
        self.init_model(weight_path, model_identity, h)
        
        self.premodule  =  PreModelModule(modelmodule_config[ 'premodel_config'])
        self.postmodule = PostModelModule(modelmodule_config['postmodel_config'], h['dataloader_config']['stft_config']['n_mel'])
    
    def init_model(self, weight_path, model_identity, h, device='cuda'):
        # import model code and load/init the model
        Model = getattr(__import__(f'TSHP.models.{model_identity}.model', fromlist=["Model"]), "Model")
        self.model, _ = Model.load_model(weight_path, h=h)
        self.model.to(device)
        w.print1(f'Loaded model : model.secpr = {self.model.secpr}')
    
    def forward(self, batch, loss_weights):
        batch = self.premodule(batch)
        batch = self.model(loss_weights, batch)
        batch = self.postmodule(batch)
        return batch
    
    def eval_infer(self, batch, loss_weights):
        batch = self.premodule(batch)
        batch = self.model.eval_infer(loss_weights, batch)
        batch = self.postmodule(batch)
        return batch
    
    def __call__(self, batch, loss_weights):
        return self.forward(batch, loss_weights)

if __name__ == '__main__':
    # test TrainModelModule
    
    # init TrainModelModule
    modelmodule_config = {
        'premodel_config': {
            # empty right now, will probably have stuff later
        },
        'postmodel_config': {
            'hcp2mel_path': None,
            'hcp2frp_path': None,
            'frp2mel_path': None,
            'textctc_path': None,
            'spkrsim_path': None,
            'vocoder_path': None,
        },
    }
    weight_path = ''
    model_identity = 'mock.tacotron2'
    h = {
        'n_rank': 1,
        'dataloader_config': {
            'stft_config': {
                'n_mel': 160,
                'hop_len': 512,
            },
        },
        'model_config': {
            'n_speakers': 384,
            'n_symbols': 256,
            'bottleneck_dim': 8,
        },
    }
    modelmodule = TrainModelModule(modelmodule_config, weight_path, model_identity, h)
    
    # run TrainModelModule.__call__()
    batch = {
        'audiopath': ['1.wav', '2.wav'],
        'gt_mel': torch.randn(2, 640, 160),
        'mel_lens': torch.tensor([640, 320]).view(-1, 1, 1),
    }
    loss_weights = {
        'mel_MAE': 0.0,
        'mel_MSE': 1.0,
    }
    batch = modelmodule(batch, loss_weights)
    assert batch['loss_dict']['mel_MSE'] == batch['loss_g_reduced'] # check loss_weights are working
    
    # TODO: add test for Vocoder
    # TODO: add test for Prosody -> Mel
    
    w.print1('TrainModelModule Test Completed!')


class DistDataloaderModule(torch.utils.data.DataLoader):
    """
    Wrapper for pytorch dataloader.
    adds kwargs;
    - loop_forever: bool == will continue looping over the dataloader forever
    - save_outputs: bool == will save outputs to RAM and return saved values instead of calling dataloader after the first dataloader loop.
    """
    def __init__(self, *args, **kwargs):
        self.loop_forever = kwargs.pop('loop_forever', False)
        self.save_outputs = kwargs.pop('save_outputs', False)
        super().__init__(*args, **kwargs)
    
    def cycle_saved(self, iterable): # taken from https://docs.python.org/3.8/library/itertools.html#itertools.cycle and modified with deepcopy
        # cycle('ABCD') --> A B C D A B C D A B C D ...
        saved = []
        for element_orig in iterable:
            element = deepcopy(element_orig)# deepcopy output so multiprocessing connections can close properly
            del element_orig
            yield element
            saved.append(element)
        while saved:
            for element in saved:
                  yield element
    
    def cycle(self, iterable):
        while True:
            for element_orig in iterable:
                element = deepcopy(element_orig)
                yield element
    
    def __iter__(self):
        if self.save_outputs:
            if self.loop_forever:
                return self.cycle_saved(super().__iter__())
            else:
                raise NotImplementedError('save_outputs without loop_forever is not implemented')
        else:
            if self.loop_forever:
                return self.cycle(super().__iter__())
            else:
                return super().__iter__()

def get_schedule_scalar(step, step_since_start, name, **conf):
    step_since_start = step_since_start + conf.get('step_offset', 0)
    
    # {'decay_step', half_life}
    if name == 'exponential_decay':
        decay_step = max(step - conf['decay_step'], 0)
        learning_rate_scalar = 1/(2**(decay_step/conf['half_life']))
    
    # {'decay_step', 'decay_interval', 'decay_scale'}
    elif name == 'step_decay':
        learning_rate_scalar = conf['decay_scale']**((step - conf['decay_step'] + conf['decay_interval']) // conf['decay_interval'])
        learning_rate_scalar = min(learning_rate_scalar, 1.0)
    
    # {'decay_start_step', 'decay_end_step'}
    elif name == 'cosine_decay':
        if step < conf['decay_start_step']:
            learning_rate_scalar = 1.0
        elif step >= conf['decay_end_step']:
            learning_rate_scalar = 0.0
        else:
            decay_step = max(step - conf['decay_start_step'], 0)
            period = conf['decay_start_step']-conf['decay_end_step']
            learning_rate_scalar = (math.cos((math.pi*decay_step)/period)+1.0)*0.5
    
    # {'start_step', 'cycle_period'}
    elif name == 'triangle_cycle':
        decay_step = max(step - conf['start_step'], 0)
        learning_rate_scalar = abs(((decay_step/conf['cycle_period']) % 1)-0.5)*2.0
    
    # {'warmup_steps'}
    elif name == 'linear_warmup':
        learning_rate_scalar = min(max(step_since_start / conf['warmup_steps'], 0.0), 1.0)
    
    # {'warmup_steps'}
    elif name == 'cosine_warmup':
        if step_since_start < conf['warmup_steps']:
            learning_rate_scalar = (math.cos((math.pi * (step_since_start - conf['warmup_steps'])) / conf['warmup_steps']) + 1.0) * 0.5
        else:
            learning_rate_scalar = 1.0
    
    # {'warmup_steps', 'start_scale'}
    elif name == 'exponential_warmup':
        if step_since_start < conf['warmup_steps']:
            log10_scale = math.log10(float(conf['start_scale']))
            learning_rate_scalar = 10**(log10_scale * (-step_since_start / conf['warmup_steps'] + 1))
        else:
            learning_rate_scalar = 1.0
    
    else:
        raise NotImplementedError(f"'{name}' is not valid learning rate scheduler")
    return learning_rate_scalar

def _maybe_create_config_files(base_directory, model_identity, run_path):
    model_config_dir = os.path.join(os.path.split(__file__)[0], '..', 'configs', *model_identity.split('.'))
    
    # if repo config is newer than users config, replace the config file with the repo default
    user_config = os.path.join(run_path, 'config.yaml')
    repo_config = os.path.join(model_config_dir, 'config.yaml')
    if not os.path.exists(user_config) or os.path.getmtime(user_config) < os.path.getmtime(repo_config):
        if os.path.exists(user_config):
            w.print1(f'moved: \"{user_config}\" -> \"{user_config + ".bak"}\"')
            shutil.move(user_config, user_config + '.bak')
        w.print1(f'copied: "{repo_config}" -> "{user_config}"')
        shutil.copyfile(repo_config, user_config)
    
    user_config_live = os.path.join(run_path, 'config_live.yaml')
    repo_config_live = os.path.join(model_config_dir, 'config_live.yaml')
    if not os.path.exists(user_config_live) or os.path.getmtime(user_config_live) < os.path.getmtime(repo_config_live):
        if os.path.exists(user_config_live):
            w.print1(f'moved: \"{user_config_live}\" -> \"{user_config_live + ".bak"}\"')
            shutil.move(user_config_live, user_config_live + '.bak')
        w.print1(f'copied: "{repo_config_live}" -> "{user_config_live}"')
        shutil.copyfile(repo_config_live, user_config_live)
    
    repo_default_config = os.path.join(os.path.split(__file__)[0], '..', 'configs', 'default_config.yaml')
    user_default_config = os.path.join(base_directory, 'default_config.yaml')
    if not os.path.exists(user_default_config) or os.stat(user_default_config).st_size == 0 or os.path.getmtime(user_default_config) < os.path.getmtime(repo_default_config):
        if os.path.exists(user_default_config):
            w.print1(f'moved: \"{user_default_config}\" -> \"{user_default_config + ".bak"}\"')
            shutil.move(user_default_config, user_default_config + '.bak')
        w.print1(f'copied: "{repo_default_config}" -> "{user_default_config}"')
        shutil.copyfile(repo_default_config, user_default_config)
    
    backup_user_default_config = os.path.join(run_path, 'default_config.yaml.bak')
    if not os.path.exists(backup_user_default_config) or os.path.getmtime(backup_user_default_config) < os.path.getmtime(user_default_config):
        w.print1(f'copied: "{user_default_config}" -> "{backup_user_default_config}"')
        shutil.copyfile(user_default_config, backup_user_default_config)

from tqdm import tqdm


def maybe_download(dataset_onlinepath):
    if dataset_onlinepath.count('@') > 0 and dataset_onlinepath.count('@')%2==0:
        dataset_path, *online_params = dataset_onlinepath.split('@')
        download_complete_path = os.path.join(dataset_path, '.download_complete')
        if not os.path.exists(download_complete_path):
            for id, site in zip_equal(online_params[::2], online_params[1::2]):
                download_unknown(dataset_path, id.strip(), site.strip())
        open(download_complete_path, 'w')
        return dataset_path
    else:
        return dataset_onlinepath


class LocalTrainModule:
    """
    Used during training.  \n
    Manages one GPU, one model and one dataloader.  \n
    Also has optional logging and checkpointing capabilities.  \n
    1) ...
    """
    def __init__(self, model_identity, run_name, weight_name, active_dataset=None, rank=0, n_rank=1, override_config: Optional[dict]=None):
        if True: # basic error messages / checking
            assert active_dataset is None or active_dataset in run_name, f'{active_dataset} missing from run_name. Add "{active_dataset}" to "{run_name}"'
            assert n_rank >= 1, f'got n_rank = {n_rank},  n_rank is less than 1'
            assert rank < n_rank, f'GPU rank of {rank} is higher than n_rank of {n_rank}'
        
        self.override_config = override_config
        self.rank = rank
        self.n_rank = n_rank
        self.h: dict = {'model_identity': model_identity, 'run_name': run_name, 'n_rank': n_rank, 'rank': rank}
        
        torch.cuda.set_device(self.rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{self.rank}'
        
        ''' misc '''
        self.save_states = {}
        self.metric_dl_keys = {'audiopath', 'basepath', 'sr', 'spkrname', 'spkr_ids', 'arpa',
                               'text', 'text_raw', 'text_ids', 'g_text_ids', 'p_text_ids', 'text_symbols', 'moji_embed', 'bert_embed',
                               'txt_lens', 'g_txt_lens', 'p_txt_lens', 'mel_lens', 'wav_lens', 'sec_lens'}
        self.model_is_allreduced = False
        
        ''' get paths for important directories and files '''
        base_directory = os.getcwd()
        #with specific_rank_first(rank_to_go_first=0, cur_rank=self.rank, n_rank=self.n_rank, timeout=1e5):
        w.print0(f'base directory: {base_directory}')
        run_path, weight_path = find_weight_path(base_directory, model_identity, run_name, weight_name)
        self.run_path = run_path
        
        if override_config is not None:
            warnings.warn('WARNING: Using override_config will disable config_live.yaml')
            self.h.update(override_config)
        else:
            ''' copy config from repo if missing '''
            _maybe_create_config_files(base_directory, model_identity, run_path)
            time.sleep(10.0)
            
            ''' load configs and merge (default, model, model live) '''
            self.update_config(os.path.join(base_directory, 'default_config.yaml')) # config that every model shares
            self.update_config(os.path.join(run_path      , 'config.yaml'        )) # config for just this model
            self.update_config(os.path.join(run_path      , 'config_live.yaml'   )) # config that can be updated in real time
        
        if active_dataset is None:
            active_dataset = self.h['dataset_config']['default_active_dataset']
        assert active_dataset in run_name, f'{active_dataset} missing from run_name'
        w.print1(f"{self.rank}: Loaded static configs")
        
        ''' (maybe) initialize distributed GPU group '''
        self._init_distributed(self.h["dist_config"], n_rank, rank)
        
        ''' load dataset '''
        self.spkrlist = None
        self.all_dictlist = None
        self.train_dictlist = None
        self.val_dictlist   = None
        self.test_dictlist  = None
        self._get_dictlists(self.h['dataset_config'], self.h['dataset_split_config'], active_dataset=active_dataset)
        w.print1(f"{self.rank}: Loaded dataset")
        
        ''' load modelmodule '''
        self.modelmodule: TrainModelModule
        self._get_modelmodule(self.h['modelmodule_config'], weight_path, model_identity, self.h)
        w.print1(f"{self.rank}: Loaded modelmodule")
        
        ''' (maybe) load metric module '''
        self.metricmodule: MetricModule
        self._get_metricmodule(self.h['metricmodule_config'])
        w.print1(f"{self.rank}: Loaded metricmodule")
        
        ''' load dataloader '''
        self.dataloader: DataLoaderModule
        self._get_dataloader(self.h['dataloader_config'])
        w.print1(f"{self.rank}: Loaded dataloadermodule")
        
        ''' load optimizer '''
        self.optimizermodule: OptimizerModule
        self._get_optimizer(self.h['optimizer_config'])
        w.print1(f"{self.rank}: Loaded optimizermodule")
        
        ''' move dataloader into workers '''
        w.print1(f"{self.rank}: Loading dataloader workers")
        self.total_batch_size: int = 0
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self._get_distdataloader(self.h['ddl_config'], self.h['segment_config'])
        self.set_cuda_benchmark_mode()
        w.print1(f"{self.rank}: Loaded dataloader workers")
        
        ''' ensure everything is loaded '''
        if weight_path is not None:
            try:
                self.load(os.path.split(os.path.splitext(weight_path)[0])[-1])
            except RuntimeError: # ignore is RuntimeError('Error loading state_dict for Model{...}')
                pass
        
        ''' and shared between GPUs '''
        self.grad_allreduce_model()
    
    def set_cuda_benchmark_mode(self):
        try:
            filelist_min_frames = self.h['dataset_config']['filelist_config']['min_dur'] * (self.h['dataloader_config']['audio_config']['sr']/self.h['dataloader_config']['stft_config']['hop_len'])
            torch.backends.cudnn.benchmark = bool(self.h['segment_config']['segment_size'] < filelist_min_frames)
        except Exception as ex:
            w.print4exc(ex)
    
    def _init_distributed(self, dist_config, n_rank, rank):
        if n_rank <= 1: return
        assert torch.cuda.is_available(), "Distributed mode requires CUDA."
        w.print1(f"{self.rank}: Initializing Distributed World")
        
        # Set cuda device so everything is done on the right GPU.
        torch.cuda.set_device(rank % torch.cuda.device_count())
        
        # Initialize distributed communication
        import datetime
        dist.init_process_group(
            backend=dist_config["dist_backend"], init_method=dist_config["dist_url"],
            world_size=n_rank, rank=rank, timeout=datetime.timedelta(seconds=600))
        
        w.print1(f"{self.rank}: Done initializing distributed world")
    
    def update_config(self, config_path):
        assert os.path.exists(config_path), f'"{config_path}" does not exist'
        try:
            with open(config_path, 'r') as f:
                conf_str = f.read()
            conf = yaml.safe_load(conf_str)
            assert conf is not None, f'"{config_path}" is empty'
            deepupdate_dicts(self.h, conf)
        except Exception:
            log.exception(f"unable to load {os.path.split(config_path)[-1]}!")
        if hasattr(self, 'optimizermodule'):
            self.optimizermodule.update_config(self.h['optimizer_config'])
    
    def _get_metricmodule(self, metricmodule_config):
        self.metricmodule = MetricModule(metricmodule_config, self.h, self.run_path, self.rank, self.n_rank)
    
    def _get_modelmodule(self, modelmodule_config, weight_path, model_identity, h):
        self.modelmodule = TrainModelModule(modelmodule_config, weight_path, model_identity, h)
        self.modelmodule.model.spkrlist = self.spkrlist
        self.modelmodule.model.textlist = self.textlist
    
    def _get_optimizer(self, optimizer_config, init_amp=True):
        self.optimizermodule = OptimizerModule(optimizer_config, model=self.modelmodule.model)
        if init_amp:
            self._initialize_mixed_precision()
    
    def _initialize_mixed_precision(self):
        # skip if amp_level is 0
        if self.h['optimizer_config']['amp_level'] == 0:
            return
        
        # convert model to mixed precision
        assert self.modelmodule is not None
        optimizers = [self.optimizermodule.optimizer_g,]
        if hasattr(self.optimizermodule, 'optimizer_d') and self.optimizermodule.optimizer_d is not None:
            optimizers.append(self.optimizermodule.optimizer_d)
        self.modelmodule.model, optimizers = amp.initialize(self.modelmodule.model, optimizers, cast_model_type=False, opt_level=f"O{self.h['optimizer_config']['amp_level']}", keep_batchnorm_fp32=False)
        self.optimizermodule.optimizer_g = optimizers[0]
        if hasattr(self, 'optimizer_d') and self.optimizermodule.optimizer_d is not None:
            self.optimizermodule.optimizer_d = optimizers[1]
        
        if self.h['optimizer_config']['amp_level'] >= 2:
            self.modelmodule.model.generator.half()
            if hasattr(self.modelmodule.model, 'discriminator'):
                self.modelmodule.model.discriminator.half()
    
    def _get_dictlists(self, dataset_config, dataset_split_config, active_dataset=None):
        """Returns trainlist, vallist and testlist"""
        
        # get dictlist
        dataset_onlinepath = dataset_config['datasets'][active_dataset] # 'filepath' or 'filepath@id@site'G:\TwiBot\TSHP\TSHP\modules\train.py
        
        with specific_rank_first(rank_to_go_first=0, cur_rank=self.rank, n_rank=self.n_rank, timeout=1e5):
            dataset_path = maybe_download(dataset_onlinepath)
            
            assert os.path.exists(os.path.abspath(dataset_path)), f'dataset does not exist at "{os.path.abspath(dataset_path)}"' 
            w.print1(f'{self.rank}: Loading {active_dataset} Filelist from "{dataset_path}"')
            self.all_dictlist, self.spkrlist = filelist(dataset_config['filelist_config']).get_multidataset_filelist(dataset_path)
            self.h['model_config']['n_speakers'] = len(self.spkrlist)
            self.textlist = [[x, i] for i, x in enumerate(['_', *self.h['dataloader_config']['text_config']['letters'], *self.h['dataloader_config']['text_config']['punctuation'], *['@'+s for s in self.h['dataloader_config']['text_config']['arpabet_symbols']]])]
            self.h['model_config']['n_symbols'] = len(self.textlist)
        
        # shuffle and split list
        random.Random(dataset_split_config['seed']).shuffle(self.all_dictlist)
        dictlists = filelist.split_filelist(self.all_dictlist, **dataset_split_config)
        self.train_dictlist = dictlists[0]
        self.val_dictlist   = dictlists[1]
        self.test_dictlist  = dictlists[2]
    
    def data_keys(self):
        # get model args
        return set(self.modelmodule.model.get_kwargs()).union(self.metric_dl_keys)
    
    def _get_dataloader(self, dataloader_config):
        self.dataloader = DataLoaderModule(
            self.data_keys(),
            dataloader_config=dataloader_config,
            spkrlist=self.spkrlist,
            training=True,
        )
        return self.dataloader
    
    def _get_max_batchsize(self, ideal_multiple=8, is_learning=True):
        # 1 - Get largest file(s) in dataset
        dataset = DatasetModule(self.dataloader, training=True, segment_config=self.h['segment_config'], dictlist=self.all_dictlist)
        maxdur_item = dataset.get_largest_od(key='dur')   # get longest duration audio file in trainset
        maxtxt_item = dataset.get_largest_od(key='quote') # get longest transcribed file in trainset
        
        # get worst case scenario max batch size
        # (where batch has both longest duration audio file and longest transcribed file at the same time)
        high = self._get_max_batchsize_for_file([maxdur_item, maxtxt_item], is_learning=is_learning)
        if high > ideal_multiple:
            high = (high//ideal_multiple)*ideal_multiple
        w.print1(f'{self.rank}: Using batch_size: {high}')
        return high
    
    def _get_max_batchsize_for_file(self, item, pre_allocate=False, max_batchsize=2, is_learning=True):
        w.print1(f"{self.rank}: Searching for maximum batch size:")
        self.save_state('batch_size_search')
        orig_n_gpus = self.modelmodule.model.h['n_rank']
        self.modelmodule.model.h['n_rank'] = 1 # run model without cross-gpu communication
        
        self.modelmodule.model.train()
        
        def test_batchsize(trainmodule, item, batchsize):
            torch.cuda.empty_cache()  # clear cache before every attempt to ensure contiguous/good memory allocation
            if type(item) in [list, tuple]:
                batch = Collate()((item * batchsize)[:batchsize])
            else:
                batch = Collate()([item, ] * batchsize)
            trainmodule.model_step(batch, is_learning=is_learning, is_logging=False, epoch_size=float('inf'), test_run=True, max_learning_rate=0.0)
        
        # 2 - Keep doubling batch size till out-of-memory error
        while True:
            try:
                test_batchsize(self, item, max_batchsize)
            except RuntimeError as ex:
                if str(ex).startswith('CUDA out of memory.'):
                    w.print0(f'{self.rank}: batch size: {max_batchsize} : FAILED')
                    break  # if OOM crash; break loop
                else:
                    w.print4exc(ex)
                    raise
            # else; double batch size and go again
            w.print0(f'{self.rank}: batch size: {max_batchsize} : PASSED')
            max_batchsize = max_batchsize * 2
        
        # 3 - Binary Search between max_batchsize//2 and max_batchsize for highest batch_size without OOM
        low = max_batchsize // 2
        high = max_batchsize - 1
        while low != high:
            mid_batch_size = math.ceil((low + high) / 2)  # get midpoint of range (rounding up)
            try:
                test_batchsize(self, item, mid_batch_size)
            except RuntimeError as ex:
                if str(ex).startswith('CUDA out of memory.'):
                    # if OOM crash; we know max_batchsize is lower than current batch_size
                    w.print0(f'{self.rank}: batch size: {mid_batch_size} : FAILED')
                    high = mid_batch_size - 1
                else:
                    w.print4exc(ex)
                    raise
            else:
                # if no crash, we know max_batchsize is higher or equal to than current batch_size
                w.print0(f'{self.rank}: batch size: {mid_batch_size} : PASSED')
                low = mid_batch_size
        assert low == high
        assert high >= 1, f'cannot find batch_size without OOM error'
        w.print1(f'{self.rank}: batch_size: {high} : found as highest safe value')
        torch.cuda.empty_cache()
        
        # reduce batch_size as a safety margin for cases like loading checkpoints or loading metric models where small amounts of additional VRAM are still needed.
        high = int(high * 0.8)
        
        # run model final time with max batch size to ensure memory is allocated correctly
        if pre_allocate:
            test_batchsize(self, item, high)
        
        self.modelmodule.model.h['n_rank'] = orig_n_gpus# re-enable cross-gpu communication
        self.load_state('batch_size_search')
        self.remove_state('batch_size_search')
        dist_barrier(self.n_rank) # check all GPUs reached the end of batch_size search
        return high
    
    def _get_distdataloader(self, ddl_config, segment_config):
        kwargs = {
            'batch_size': self._get_max_batchsize() if 'max' in str(ddl_config['batch_size']) else int(ddl_config['batch_size']),
            'num_workers': os.cpu_count()//4 if ddl_config['num_workers'] == 'auto' else ddl_config['num_workers'],
            'collate_fn': Collate(),
        }
        trainset = DatasetModule(self.dataloader, training=True, segment_config=segment_config, dictlist=self.train_dictlist)
        sampler = DistributedSampler(trainset) if self.n_rank > 1 else None
        self.train_dataloader = DistDataloaderModule(trainset, sampler=sampler, persistent_workers=ddl_config['caching_level'] == 1, **kwargs, shuffle=(sampler is None))
        
        loop_val_forever = True
        save_val_forever = True
        valset = DatasetModule(self.dataloader, training=False, segment_config=segment_config, dictlist=self.val_dictlist)
        sampler = DistributedSampler(valset, shuffle=False) if self.n_rank > 1 else None
        self.val_dataloader   = DistDataloaderModule(valset, sampler=sampler, persistent_workers=ddl_config['caching_level'] == 1, **kwargs, loop_forever=loop_val_forever, save_outputs=save_val_forever)
        
        testset = DatasetModule(self.dataloader, training=False, segment_config=segment_config, dictlist=self.test_dictlist)
        sampler = DistributedSampler(testset, shuffle=False) if self.n_rank > 1 else None
        self.test_dataloader  = DistDataloaderModule(testset, sampler=sampler, persistent_workers=ddl_config['caching_level'] == 1, **kwargs)
    
    def _get_alldistdataloader(self, ddl_config, segment_config, arpabet):
        kwargs = {
            'batch_size': self._get_max_batchsize(is_learning=False)//2 if 'max' in str(ddl_config['batch_size']) else int(ddl_config['batch_size']),
            'num_workers': os.cpu_count() if ddl_config['num_workers'] == 'auto' else ddl_config['num_workers'],
            'collate_fn': Collate(),
        }
        allset = DatasetModule(self.dataloader, training=True, segment_config=segment_config, dictlist=self.all_dictlist)
        allset.data_loader.p_arpabet = bool(arpabet)
        sampler = DistributedSampler(allset, shuffle=False) if self.n_rank > 1 else None
        self.all_dataloader = DistDataloaderModule(allset, sampler=sampler, persistent_workers=ddl_config['caching_level'] == 1, **kwargs, shuffle=False)
    
    def grad_allreduce_model(self):
        # (maybe) add batch_size weighted gradient syncronization to model
        batch_size = self.train_dataloader.batch_size
        if self.n_rank > 1:
            w.print1(f'{self.rank}: applying grad_allreduce to model with {self.n_rank} gpus')
            self.total_batch_size = dist_add(torch.tensor(batch_size), async_op=False)
            w.print1(f'{self.rank}: got total_batch_size: {self.total_batch_size}')
            weight = (batch_size/self.total_batch_size)*dist.get_world_size()
            self.modelmodule.model = apply_gradient_allreduce(weight)(self.modelmodule.model)
            self.model_is_allreduced = True
        else:
            self.total_batch_size = int(batch_size)
    
    def get_max_learning_rate(self, start_learning_rate: float, end_learning_rate: float, secpr_till_end: float, file_window_size: int = 8192, reset_state_when_finished: bool = False):
        learning_rate_list, total_loss_list = self._get_max_learning_rate_search(
            start_learning_rate, end_learning_rate, reset_state_when_finished, secpr_till_end)
        
        max_learning_rate = self._get_max_learning_rate_from_losslr_list(
            file_window_size, learning_rate_list, total_loss_list) * 0.5
        
        # if n_rank > 1:
        if self.n_rank > 1:
            w.print1(f'{self.rank}: found max_learning_rate: {max_learning_rate}')
            max_learning_rate = dist_mean(max_learning_rate, dist.get_world_size()) 
        
        # return that learning_rate
        w.print1(f'{self.rank}: using max_learning_rate: {max_learning_rate}')
        self.modelmodule.model.max_learning_rate.fill_(max_learning_rate)
        return max_learning_rate
    
    def _get_max_learning_rate_search(self, start_learning_rate: float, end_learning_rate: float, reset_state_when_finished: bool, secpr_till_end: float, use_tqdm=False, finish_at_current_secpr=True):
        w.print1(f'{self.rank}: Finding max learning rate...')
        init_secpr = self.modelmodule.model.secpr.item()
        
        # temporarily change learning rate schedule
        orig_schedule = deepcopy(self.h['optimizer_config']['schedulers'])
        self.h['optimizer_config']['schedulers'] = [{
            'name'        : 'exponential_warmup',
            'start_scale' : start_learning_rate / end_learning_rate,
            'warmup_steps': secpr_till_end,
            'step_offset' : secpr_till_end-init_secpr if finish_at_current_secpr else -init_secpr,
        }, ]
        orig_grad_g_tracker = self.optimizermodule.grad_g_tracker
        orig_grad_d_tracker = self.optimizermodule.grad_d_tracker
        self.optimizermodule.grad_g_tracker = MovingPercentile(orig_grad_g_tracker.percentile, 8)
        self.optimizermodule.grad_d_tracker = MovingPercentile(orig_grad_d_tracker.percentile, 8)
        self.optimizermodule.grad_g_clip_thresh = float('inf')
        self.optimizermodule.grad_d_clip_thresh = float('inf')
        orig_secpr = self.modelmodule.model.secpr.item()
        if finish_at_current_secpr:
            self.modelmodule.model.secpr -= secpr_till_end
        if reset_state_when_finished:
            self.save('checkpoint_0')
        
        # run model with exponentially increasing learning rate
        dl_len = len(self.train_dataloader)
        smoothed_loss = None
        best_smoothed_loss = float('inf')
        total_loss_list = []
        learning_rate_list = []
        tqdm_obj = None
        if use_tqdm:
            tqdm_obj = tqdm(desc='finding learning rate', total=secpr_till_end)
        counter = 0
        cont = True
        start_data_time = time.time()
        while cont:
            for batch in self.train_dataloader:
                before_secpr = self.modelmodule.model.secpr.item()
                assert math.isfinite(before_secpr)
                try:
                    batch, learning_rate = self.model_step(
                        batch,
                        is_learning=True,
                        is_logging=not reset_state_when_finished,
                        plot_prepend='train',
                        epoch_size=dl_len,
                        dataload_start_time=start_data_time,
                        max_learning_rate=end_learning_rate,
                        load_live_config=False,
                    )
                    if use_tqdm:
                        tqdm_obj.update(round(self.modelmodule.model.secpr.item() - before_secpr))
                except ValueError:
                    break
                total_loss = batch['loss_g_reduced'] + batch.get('loss_d_reduced', 0.0)
                
                if math.isfinite(total_loss):
                    total_loss_list.append(total_loss)
                    learning_rate_list.append(learning_rate)
                    w.print0(f'learning_rate search at iter: {counter} secpr: {self.modelmodule.model.secpr} lr: {self.optimizermodule.learning_rate:.1e}')
                    
                    batch_size = next(t.shape[0] for k, t in batch.items() if type(t) is torch.Tensor and t.dim() >= 3)
                    if not reset_state_when_finished:
                        smoothing_factor = 0.99 ** batch_size
                        if smoothed_loss is None:
                            smoothed_loss = total_loss
                        else:
                            smoothed_loss = (smoothed_loss*smoothing_factor) + (total_loss*(1.-smoothing_factor))
                            
                        if smoothed_loss < best_smoothed_loss:
                            best_smoothed_loss = smoothed_loss
                            self.save_state('lr_best_loss', logging_level=0)
                
                del batch
                
                # stop when loss values start increasing rapidly
                # [INSERT STOP CODE HERE]
                if (smoothed_loss or total_loss) > 1000.0 or not math.isfinite(total_loss):
                    w.print1('Loss reached threshold, ending LR search')
                    cont = False
                    break
                loss_scale = amp._amp_state.loss_scalers[0].loss_scale() if self.h['optimizer_config']['amp_level'] > 0 else None
                if loss_scale is not None and loss_scale < 8:
                    w.print1('loss_scale dropped below safe level, ending LR search')
                    cont = False
                    break
                if self.modelmodule.model.secpr.item() >= (init_secpr if finish_at_current_secpr else secpr_till_end):
                    w.print1('Model secpr reached threshold, ending LR search')
                    cont = False
                    break
                start_data_time = time.time()
                counter += 1
        if use_tqdm:
            tqdm_obj.close()
        w.print0(list(zip(total_loss_list, learning_rate_list)))
        if reset_state_when_finished:
            self.load('checkpoint_0')
        else:
            self.load_state('lr_best_loss')
            self.remove_state('lr_best_loss')
        if finish_at_current_secpr:
            self.modelmodule.model.secpr.fill_(orig_secpr)
        # revert learning rate schedule to original
        self.h['optimizer_config']['schedulers'] = orig_schedule
        self.optimizermodule.grad_g_tracker = orig_grad_g_tracker
        self.optimizermodule.grad_d_tracker = orig_grad_d_tracker
        return learning_rate_list, total_loss_list
    
    def _get_max_learning_rate_from_losslr_list(self, file_window_size, learning_rate_list: list, total_loss_list: list):
        # find learning_rate when loss value was at it's minimum
        total_batch_size = dist_add(self.train_dataloader.batch_size, self.n_rank)
        window_size = math.ceil(file_window_size / total_batch_size)
        window_means = []
        window_indexes = []
        window_sum = sum(total_loss_list[:window_size - 1])
        for end_i, _ in list(enumerate(total_loss_list))[window_size:]:
            start_i = end_i - window_size
            mid_index = (start_i + end_i) // 2
            if math.isfinite(total_loss_list[end_i]):
                window_sum += total_loss_list[end_i]
            if math.isfinite(total_loss_list[start_i]):
                window_sum -= total_loss_list[start_i]
            
            window_means.append(window_sum / window_size)
            window_indexes.append(mid_index)
        max_learning_rate = learning_rate_list[window_indexes[window_means.index(min(window_means))]]
        return max_learning_rate
    
    def save(self, save_name, logging_level=1):
        """Save model+optimizer to disk"""
        if self.rank != 0: # skip if not GPU 0
            return
        
        os.makedirs(os.path.join(self.run_path, 'weights'), exist_ok=True)
        wpath = os.path.join(self.run_path, 'weights'   , save_name+'.ptw')
        self.modelmodule.model.save_model(wpath)
        getattr(w, f'print{logging_level}')(f'Saved TrainModule to "{wpath}"')
        
        os.makedirs(os.path.join(self.run_path, 'optimizers'), exist_ok=True)
        opath = os.path.join(self.run_path, 'optimizers', save_name+'.pto')
        self.optimizermodule.save(opath)
        
        os.makedirs(os.path.join(self.run_path, 'metricmodule'), exist_ok=True)
        mpath = os.path.join(self.run_path, 'metricmodule', save_name+'.ptm')
        self.metricmodule.save(mpath)
    
    def load(self, save_name):
        wpath = os.path.join(self.run_path, 'weights'   , save_name+'.ptw')
        self.modelmodule.model.load_state_dict(torch.load(wpath, map_location='cuda')['state_dict'], strict=False)
        w.print1(f'{self.rank}: Loaded TrainModule from "{wpath}"')
        
        opath = os.path.join(self.run_path, 'optimizers', save_name+'.pto')
        if os.path.exists(opath) and getattr(self, 'optimizermodule', None) is not None:
            self.optimizermodule.load(opath)
        else:
            w.print2(f'{self.rank}: "{opath}" does not exist')
        
        mpath = os.path.join(self.run_path, 'metricmodule', save_name+'.ptm')
        if os.path.exists(mpath) and getattr(self, 'metricmodule', None) is not None:
            self.metricmodule.load(mpath)
        else:
            w.print2(f'{self.rank}: "{mpath}" does not exist')
    
    def save_state(self, key, device='cpu', logging_level=1):
        """Save model+optimizer to RAM"""
        if self.metricmodule.epochavg_dict:
            assert isinstance(next(iter(next(iter(self.metricmodule.epochavg_dict.values())).values())), MovingDictAverage)
        getattr(w, f'print{logging_level}')(f'Saving model state to RAM with key: {key}')
        state = [deepcopy(deepto(self.modelmodule.model.state_dict(), device))]
        if self.optimizermodule.optimizer_g is not None:
            state.append(deepcopy(deepto(self.optimizermodule.optimizer_g.state_dict(), device)))
        if self.optimizermodule.optimizer_d is not None:
            state.append(deepcopy(deepto(self.optimizermodule.optimizer_d.state_dict(), device)))
        if self.metricmodule is not None:
            state.append(deepcopy(deepto(self.metricmodule._get_state(), device)))
        self.save_states[key] = state
    
    def load_state(self, key):
        w.print1(f'{self.rank}: Loading model state with key: {key}')
        state = self.save_states[key]
        self.modelmodule.model.load_state_dict(deepto(state.pop(0), 'cuda'))
        if self.optimizermodule.optimizer_g is not None:
            self.optimizermodule.optimizer_g.load_state_dict(deepto(state.pop(0), 'cuda'))
        if self.optimizermodule.optimizer_d is not None:
            self.optimizermodule.optimizer_d.load_state_dict(deepto(state.pop(0), 'cuda'))
        if self.metricmodule.epochavg_dict:
            assert isinstance(next(iter(next(iter(self.metricmodule.epochavg_dict.values())).values())), MovingDictAverage)
        if self.metricmodule is not None:
            self.metricmodule.load_state(state.pop(0))
        if self.metricmodule.epochavg_dict:
            assert isinstance(next(iter(next(iter(self.metricmodule.epochavg_dict.values())).values())), MovingDictAverage)
    
    def remove_state(self, key):
        if key in self.save_states:
            del self.save_states[key]
    
    def _lr_schedule_step(self, patience_secpr: Optional[float]=None, lr_step_scale: Optional[float]=0.31622776601683794):
        if patience_secpr is None:
            patience_secpr = 1_000_000
        if self.modelmodule.model.secpr >= self.modelmodule.model.best_cross_val_secpr + patience_secpr:
            # if no improvement after patience_secpr time;
            
            # load best checkpoint, reduce model lr and update checkpoint lr to match
            try:
                self.load_state('best_cross_val')
            except KeyError:
                self.load('best_cross_val')
            dist_barrier(self.n_rank, assert_same_call=True)
            self.modelmodule.model.lr_multiplier.mul_(lr_step_scale)
            self.save_state('best_cross_val')
            self.save('best_cross_val')
    
    def _get_scheduled_learning_rate(self, max_learning_rate: float):
        assert max_learning_rate is not None, 'can\'t find max_learning_rate. call get_max_learning_rate() to automatically find it. (note-it can take a while to execute)'
        learning_rate = deepcopy(max_learning_rate)
        if hasattr(self.modelmodule.model, 'lr_multiplier'):
            learning_rate *= self.modelmodule.model.lr_multiplier.item()
        for scheduler_config in self.h['optimizer_config']['schedulers']:
            learning_rate *= get_schedule_scalar(self.modelmodule.model.secpr.item(), step_since_start=self.modelmodule.model.secpr.item(), **scheduler_config)
        return learning_rate
    
    def _model_step(self, batch: dict, is_learning: bool, is_logging: bool, epoch_size: int=None, dataload_start_time=None, plot_scalars=None, plot_media=None, plot_prepend=None, batch_size=None, test_run=False, call_func='__call__', max_learning_rate=None, load_live_config=True):
        if is_logging:
            assert plot_prepend is not None, 'plot_prepend (name for logging) must be specified to perform logging. examples: \'train\' and \'validation\''
        if plot_scalars is None:
            plot_scalars = is_learning
        if batch_size is None:
            batch_size = next(t.shape[0] for k, t in batch.items() if type(t) is torch.Tensor and t.dim() >= 3)
        if self.n_rank > 1 and self.model_is_allreduced:
            batch_size = dist_add(batch_size, self.n_rank)
        if max_learning_rate is None and getattr(self.modelmodule.model, 'max_learning_rate', 0.0) > 0.0:
            max_learning_rate = self.modelmodule.model.max_learning_rate
        
        loss_weights = self.h['loss_weights']
        model_start_time = time.time()
        if is_learning:
            if load_live_config and self.override_config is None:
                self.update_config(os.path.join(self.run_path, 'config_live.yaml'))
            assert epoch_size is not None, 'epoch_size missing while model is learning'
            batch = getattr(self.modelmodule, call_func)(batch, loss_weights)
            loss_g = batch['loss_g']
            loss_d = batch.get('loss_d', None)
            grad_norm_total, grad_norm_g, grad_norm_d = self.optimizermodule.step(
                self.modelmodule.model, loss_g, loss_d, batch_size, test_run=test_run,
                learning_rate=self._get_scheduled_learning_rate(max_learning_rate))
            self.modelmodule.model.offset_tracker(iteration_delta=1,
                                                  epoch_delta=dist_mean(1/epoch_size, self.n_rank if self.model_is_allreduced else 1),
                                                  secpr_delta=dist_add(batch['sec_lens'].sum().item(), self.n_rank if self.model_is_allreduced else 1))
        else:
            grad_norm_total = grad_norm_g = grad_norm_d = 0.0
            if not hasattr(self.modelmodule.model, 'discriminator'):
                grad_norm_d = None
            with torch.no_grad():
                batch = getattr(self.modelmodule, call_func)(batch, loss_weights)
        
        learning_rate: Optional[float] = self.optimizermodule.learning_rate if self.optimizermodule.optimizer_g is not None else None
        if is_logging:
            loss_scale = amp._amp_state.loss_scalers[0].loss_scale() if self.h['optimizer_config']['amp_level'] > 0 else None
            if loss_scale is not None and loss_scale < 4:
                raise RuntimeError("loss_scale is below 4 and model will destable / is unstable.")
            
            additional_scalars = {
                'grad_norm_total': grad_norm_total,
                'grad_norm_g': grad_norm_g,
                'grad_norm_d': grad_norm_d,
                'loss_scale': loss_scale,
                'learning_rate': learning_rate,
            }
            
            self.metricmodule.log_step(
                batch,
                self.modelmodule.model,
                additional_scalars,
                batch_size=batch_size,
                epoch_size=epoch_size,
                prepend=plot_prepend,
                plot_scalars=plot_scalars,
                plot_dynamic_items_media=plot_media,
                model_start_time=model_start_time,
                dataload_start_time=dataload_start_time,
                rank=self.rank)
        return batch, learning_rate
    
    def model_step(self, batch, is_learning, is_logging, epoch_size=None, dataload_start_time=None, plot_scalars=None, plot_media=None, plot_prepend=None, batch_size=None, test_run=False, call_func='__call__', max_learning_rate=None, load_live_config=True, seed=None):
        if seed is not None:
            with torch.random.fork_rng():
                torch.random.manual_seed(seed)
                return self._model_step(batch, is_learning, is_logging, epoch_size=epoch_size, dataload_start_time=dataload_start_time, plot_scalars=plot_scalars, plot_media=plot_media, plot_prepend=plot_prepend, batch_size=batch_size, test_run=test_run, call_func=call_func, max_learning_rate=max_learning_rate, load_live_config=load_live_config)
        else:
            return self._model_step(batch, is_learning, is_logging, epoch_size=epoch_size, dataload_start_time=dataload_start_time, plot_scalars=plot_scalars, plot_media=plot_media, plot_prepend=plot_prepend, batch_size=batch_size, test_run=test_run, call_func=call_func, max_learning_rate=max_learning_rate, load_live_config=load_live_config)
    
    def train_epoch_till(self, secpr=1e15, epoch=999_999_999, iteration=999_999_999, sec_elapsed=999_999_999.9, learning_rate=1e-7, start_time=None, max_learning_rate=None, patience_secpr: Optional[float]=None, val_seed: Optional[int]=None):
        if start_time is None:
            start_time = time.time()
        self.modelmodule.model.train()
        
        start_data_time = time.time()
        dl_len = len(self.train_dataloader)
        val_dl_len = len(self.val_dataloader)
        w.print1(f"{self.rank}: dl_len:{dl_len}  val_dl_len:{val_dl_len}")
        val_dataloader_iter = iter(self.val_dataloader)
        for batch in self.train_dataloader:
            # run model (safely)
            train_ms_kwargs = {
                'is_learning': True,
                'is_logging': True,
                'epoch_size': dl_len,
                'dataload_start_time': start_data_time,
                'plot_prepend': 'train',
                'max_learning_rate': max_learning_rate,
            }
            try:
                self.modelmodule.model.train()
                self.model_step(batch, **train_ms_kwargs)
            except RuntimeError as ex:
                if str(ex).startswith('CUDA out of memory.'):
                    # if OOM crash;
                    w.print4exc(ex)
                    w.print4('Ran out of memory. Saving model and retrying.')
                    # clear VRAM
                    torch.cuda.empty_cache()
                    # try again
                    try:
                        self.model_step(batch, **train_ms_kwargs,)
                    except RuntimeError: pass
                else:
                    w.print4exc(ex)
                    raise
            del batch
            
            # run cross-validation
            val_start_data_time = time.time()
            val_batch = next(val_dataloader_iter)
            val_ms_kwargs = {
                'is_learning': False,
                'is_logging': True,
                'epoch_size': val_dl_len,
                'dataload_start_time': val_start_data_time,
                'plot_scalars': True,
                'plot_prepend': 'cross_val',
                'max_learning_rate': max_learning_rate,
                'seed': val_seed if val_seed is not None else hash(val_batch['audiopath'][0]),
            }
            try:
                self.modelmodule.model.eval()
                self.model_step(val_batch, **val_ms_kwargs)
                
                # https://pastebin.com/3DndzKis
                cross_val_loss = self.metricmodule.epochavg_dict['cross_val']['loss_total_reduced_MAE'].mean()
                if cross_val_loss < self.modelmodule.model.best_cross_val and self.modelmodule.model.secpr > self.modelmodule.model.best_cross_val_secpr:
                    self.modelmodule.model.best_cross_val_secpr.fill_(self.modelmodule.model.secpr)
                    self.modelmodule.model.best_cross_val.fill_(cross_val_loss)
                    
                    # wait at least 0k secpr before saving to RAM (to reduce I/O slowdown)
                    if self.modelmodule.model.secpr.item() > getattr(self, 'last_saved_best_cross_val_to_ram_secpr', 0.0) + 0:
                        self.save_state('best_cross_val', logging_level=0)
                        self.last_saved_best_cross_val_to_ram_secpr = self.modelmodule.model.secpr.item()
                    
                    # wait at least 2k secpr before saving to disk (to reduce I/O slowdown)
                    if self.modelmodule.model.secpr.item() > getattr(self, 'last_saved_best_cross_val_to_disk_secpr', 0.0) + 10_000:
                        self.save('best_cross_val')
                        self.last_saved_best_cross_val_to_disk_secpr = self.modelmodule.model.secpr.item()
            except RuntimeError as ex:
                if str(ex).startswith('CUDA out of memory.'):
                    # if OOM crash;
                    w.print4exc(ex)
                    w.print4('Ran out of memory. Saving model and retrying.')
                    # clear VRAM
                    torch.cuda.empty_cache()
                    # try again
                    try:
                        self.model_step(val_batch, **val_ms_kwargs)
                    except RuntimeError: pass
                else:
                    w.print4exc(ex)
                    raise
            del val_batch
            
            self._lr_schedule_step(patience_secpr=patience_secpr)
            
            # latest save
            seconds_between_latest_saves = 60 * 10
            if time.time() > self.modelmodule.model.last_save_time + seconds_between_latest_saves:
                self.save('latest_epoch')
                self.modelmodule.model.last_save_time.fill_(time.time())
            
            # stop conditions
            if self.modelmodule.model.epoch.item() >= epoch:
                w.print1(f'{self.rank}: Epochs reached: {self.modelmodule.model.epoch}\nFinished training.')
                break
            if self.modelmodule.model.iteration.item() >= iteration:
                w.print1(f'{self.rank}: Iterations reached: {self.modelmodule.model.iteration}\nFinished training.')
                break
            if self.modelmodule.model.secpr.item() >= secpr:
                w.print1(f'{self.rank}: SecPr reached: {self.modelmodule.model.secpr:.1f}\nFinished training.')
                break
            if time.time() >= (start_time + sec_elapsed):
                w.print1(f'{self.rank}: Seconds Elapsed reached: {time.time() - start_time:.1f}\nFinished training.')
                break
            if self.optimizermodule.learning_rate <= learning_rate and self.modelmodule.model.iteration > 2000:
                w.print1(f'{self.rank}: Learning Rate reached: {self.optimizermodule.learning_rate:.2e}\nFinished training.')
                break
            
            start_data_time = time.time()
    
    def train_till(self, secpr=1e15, epoch=999_999_999, iteration=999_999_999, sec_elapsed=999_999_999.9, learning_rate=1e-7, max_learning_rate=None, patience_secpr: Optional[float]=None):
        start_time = time.time()
        counter = 0
        while (
                self.modelmodule.model.epoch < epoch and
                self.modelmodule.model.iteration < iteration and
                self.modelmodule.model.secpr < secpr and
                time.time() < (start_time + sec_elapsed) and
                (self.optimizermodule.learning_rate > learning_rate or counter < 1)):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(int(self.modelmodule.model.epoch.item()))
            self.train_epoch_till(secpr=secpr, epoch=epoch, iteration=iteration, sec_elapsed=sec_elapsed, learning_rate=learning_rate, max_learning_rate=max_learning_rate, patience_secpr=patience_secpr)
            
            # TODO: set epoch for distributed sampler so shuffling works properly
            counter+=1
        w.print1(f"{self.rank}: Finished Training!")
        if self.modelmodule.model.epoch >= epoch:
            w.print1(f"{self.rank}: Epochs reached target")
        if self.modelmodule.model.iteration >= iteration:
            w.print1(f"{self.rank}: Iterations reached target")
        if self.modelmodule.model.secpr >= secpr:
            w.print1(f"{self.rank}: Seconds-Processed reached target")
        if time.time() >= (start_time + sec_elapsed):
            w.print1(f"{self.rank}: Wallclock-Time reached target")
        if self.optimizermodule.learning_rate <= learning_rate:
            w.print1(f"{self.rank}: Learning-Rate reached target")
    
    def dump_features(self, pr_key='alignments', file_extension='_align.pt', arpabet=False):
        w.print1(f'{self.rank}: Dumping Features "{pr_key}" to disk with ext "{file_extension}"')
        self.modelmodule.model.eval()
        orig_n_gpus = self.modelmodule.model.h['n_rank']
        self.modelmodule.model.h['n_rank'] = 1 # run model without cross-gpu communication
        self.n_rank = 1
        
        self._get_alldistdataloader(self.h['ddl_config'], self.h['segment_config'], arpabet=arpabet)
        
        start_data_time = time.time()
        n_files = len(self.all_dataloader.dataset)
        it = 0
        dl_len = len(self.all_dataloader)
        w.print1(f"{self.rank}: dl_len:{dl_len}")
        for batch in self.all_dataloader:
            # run model (safely)
            train_ms_kwargs = {
                'is_learning': False,
                'is_logging': False,
                'epoch_size': dl_len,
                'dataload_start_time': start_data_time,
            }
            batch, _ = self.model_step(batch, **train_ms_kwargs)
            
            audiopath = batch['audiopath']
            pr_feat = batch[pr_key]
            try:
                lens = get_matched_lens(pr_feat, [t for k, t in batch.items() if 'lens' in k])
            except RuntimeWarning:
                if 'alignment' in pr_key:
                    lens = batch['txt_lens']
            
            lens2 = None
            if 'alignment' in pr_key:
                lens2 = get_matched_lens(pr_feat.transpose(1, 2), [t for k, t in batch.items() if 'lens' in k and not (k.startswith('g_') or k.startswith('p_'))])
            
            for i in range(len(audiopath)):
                writepath = os.path.splitext(audiopath[i])[0]+file_extension
                if os.path.exists(writepath):
                    w.print1(f'Overwrote "{writepath}"')
                else:
                    w.print1(f'Wrote "{writepath}"')
                pr_feati = pr_feat[i]
                if lens is not None:
                    pr_feati = pr_feati[:lens[i]]
                if 'alignment' in pr_key:
                    pr_feati = pr_feati[:, :lens2[i]]
                    assert pr_feati.shape[0] == lens[i]
                    assert pr_feati.shape[1] == lens2[i]
                safe_write(pr_feati.unsqueeze(0).detach().clone(), writepath)
                it+=1
            w.print1(f'{dist_add(it, n_rank=self.n_rank):>7}/{n_files:<7}')
        self.modelmodule.model.h['n_rank'] = orig_n_gpus
        self.n_rank = orig_n_gpus


if __name__ == '__main__':
    # test LocalTrainModule
    
    # TODO: test init
    
    # TODO: test saving/loading
    
    # TODO: test saving/loading when the model config/state_dict changes
    
    # TODO: test loading when checkpoint is in different run_path
    
    # TODO: test batch size search
    
    # TODO: test batch size with max limit
    
    # TODO: test hard-set batch size
    
    # TODO: test learning rate search
    
    # TODO: test learning rate hard-set
    
    # TODO: test model_step (with various args)
    
    # TODO: test E2E training of very small model
    
    # TODO: test feature/alignment dumping with finished model
    
    # TODO: add exception for missing config file
    
    w.print1('LocalTrainModule Test Completed!')

class GlobalTrainModule(LocalTrainModule):
    """
    Used during training.  \n
    Launches a cluster of LocalTrainModule(s) and makes them (and their GPUs) work together on a single shared model.  \n
    1) ...
    """
    def __init__(self, model_identity, run_name, weight_name, active_dataset=None, rank=0, n_rank=1, override_config: Optional[dict]=None, args=None):
        assert args is not None, 'args must not be None for multiGPU training'
        if rank == 0 and n_rank > 1:
            argslist = [[f'--{k}', f'{"" if v is None else v}'] for k, v in vars(args).items() if k != 'rank' and v != '']
            argslist = [v for x in argslist for v in x if v != '']
            if '--active_dataset' in argslist and [*argslist, '--'][argslist.index('--active_dataset')+1].startswith('--'):
                argslist.insert(argslist.index('--active_dataset')+1, '')
            if '--weight_name' in argslist and [*argslist, '--'][argslist.index('--weight_name')+1].startswith('--'):
                argslist.insert(argslist.index('--weight_name')+1, '')
            if '--lr' in argslist and [*argslist, '--'][argslist.index('--lr')+1].startswith('--'):
                argslist.insert(argslist.index('--lr')+1, 'nan')
            
            # spawn subprocesses
            self.workers = []
            for i in range(1, n_rank):
                argslist.extend([f'--rank', f'{i}'])
                #print([str(sys.executable), '-m', 'TSHP.train']+argslist)
                p = subprocess.Popen([str(sys.executable), '-m', 'TSHP.train']+argslist,)# stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)#, stdout=stdout)
                self.workers.append(p)
                argslist = argslist[:-2]
            
            super().__init__(model_identity, run_name, weight_name, active_dataset=active_dataset, rank=rank, n_rank=n_rank, override_config=override_config)
        else:
            super().__init__(model_identity, run_name, weight_name, active_dataset=active_dataset, rank=rank, n_rank=n_rank, override_config=override_config)
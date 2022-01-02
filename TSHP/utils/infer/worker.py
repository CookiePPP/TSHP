import argparse
import os
import time
import warnings
from copy import deepcopy
from glob import glob
from typing import Optional, Dict

import torch

from scipy.io.wavfile import write

from CookieSpeech.functions.load_ctts2_checkpoints import load_model_from_path
from CookieSpeech.modules.train_utils import get_all_model_refs, guess_model_ref_from_path
from CookieSpeech.utils.arg_utils import get_args
from CookieSpeech.utils.dataset.dataset import DataLoaderModule, Collate
from CookieSpeech.utils.misc_utils import zip_equal, deepupdate_dicts


class Worker:
    def __init__(self, weights_directory, device='cuda'):
        self.models = {} # {checkpoint_path: model}
        self.model_refs = {} # {checkpoint_path: model_ref}
        self.mparams = {}
        
        self.device = device
        
        self.weights_directory = weights_directory
        self.models_directory = os.path.join(__file__, '../../../models')
        self.additional_weights_directories = ["../metric_models",]
        
        self.all_coded_model_refs = get_all_model_refs()
        self.latest_h = {}
        self.all_args = set()
    
    def get_available_model_checkpoints(self):
        # browse local directories
        
        checkpoints = []
        for modeltype_name in os.listdir(self.weights_directory):
            modeltype_path = os.path.join(self.weights_directory, modeltype_name)
            for model_name in os.listdir(modeltype_path):
                model_path = os.path.join(modeltype_path, model_name)
                for run_name in os.listdir(model_path):
                    run_path = os.path.join(model_path, run_name)
                    weights_path = os.path.join(run_path, 'weights')
                    
                    for weight_name in os.listdir(weights_path):
                        if weight_name.endswith('.ptw'):
                            checkpoints.append(weight_name)
        
        for weights_directory in self.additional_weights_directories:
            for name in os.listdir(weights_directory):
                path = os.path.join(weights_directory, name)
                if os.path.isfile(path) and path.endswith('.ptw'):
                    checkpoints.append(path)
        
        assert len(checkpoints)
        return sorted(checkpoints)
    
    def load_model(self, path, device='cuda'):
        if os.path.isdir(path):
            path = get_highest_iter_checkpoint(path)
        model_ref = guess_model_ref_from_path(path)
        if path in self.models:
            return model_ref
        
        print(f"Loading {model_ref} from '{path}'")
        
        # load checkpoint
        model, h = load_model_from_path(path)
        print(f'loaded {model_ref} @ {model.iteration} iters')
        self.models[path] = model.to(device).half()
        self.model_refs[path] = model_ref
        
        # exceptions for configs
        if 'vocoder' in model_ref:
            del model.h['dataloader_config']['text_config']
        
        # merge configs/args of all currently loaded models
        self.all_args = set([arg for model in self.models.values() for arg in get_args(model.generator.infer)])
        self.all_args.add('spkrname')
        if model.h is not None:
            deepupdate_dicts(self.latest_h, model.h)
        return model_ref
    
    def maybe_load_model(self, path, device='cuda'):
        ref = self.load_model(path, device=device)
        return ref
    
    def init_dataloader(self):
        self.dataloader = DataLoaderModule(
            self.all_args, False,
            self.latest_h['dataloader_config'],
            next(iter(self.models.values())).spkrlist,
            allow_caching=False
        )
    
    def check_spkr_in_models(self, spkrname_list):
        for spkrname in spkrname_list:
            for cp_path, model in self.models.items():
                model_ref = self.model_refs[cp_path]
                if hasattr(model, 'spkrlist') and spkrname not in model.spkrlist:
                    warnings.warn(f'{spkrname} not found in {model_ref} spkrlist!')
    
    def infer_html(self, html_wd, checkpoints):
        wd = html_wd
        
        # deal with any datatype issues from html request
        
        wd = self.infer(wd, checkpoints)
        return wd
    
    def infer(self, wd, checkpoints, unload_unneeded_models=False, b_arpabet=False, batch_size=None, extra_dl_feats=None):
        if extra_dl_feats is None:
            extra_dl_feats = set()
        if batch_size is None:
            batch_size = int(2**16)
        wd = deepcopy(wd)
        for k in list(wd.keys()):
            if wd[k] is None:
                del wd[k]
        B = len(wd['spkrname'])
        
        # (maybe) unload models that are not used during this infer call
        remove_list = []
        for model_cp_path in self.models:
            if model_cp_path not in checkpoints:
                remove_list.append(model_cp_path)
        if unload_unneeded_models:
            for model_cp_path in remove_list:
                del self.models[model_cp_path]
                del self.model_refs[model_cp_path]
            if len(remove_list):
                torch.cuda.empty_cache()
        
        # load any unloaded models and track order for when they're ran
        model_refs = []
        for checkpoint in checkpoints:
            ref = self.maybe_load_model(checkpoint, device=self.device)
            model_refs.append(ref)
        
        self.all_args.update(extra_dl_feats)
        with torch.no_grad():
            # run dataloader to get input features for model
            if not hasattr(self, 'dataloader') or self.dataloader.args_raw != self.all_args:
                self.latest_h['dataloader_config']['text_config']['p_arpabet'] = b_arpabet
                self.init_dataloader()
            self.dataloader.spkrlist = next(iter(self.models.values())).spkrlist
            self.dataloader.spkrlookup = None # will become dict {spkrname: spkr_id, ...} if spkr_ids are needed
            
            start_time = time.time()
            wd_list = []
            for i in range(B):
                wd_list.append(self.dataloader.get_od(
                    wd.get('audiopath', [None,]*B)[i],
                    wd.get( 'text_raw', [None,]*B)[i],
                    wd.get( 'spkrname', [None,]*B)[i],
                ))
            print(f"{time.time() - start_time:.2f}s elapsed (dataloader)")
            
            wd_out: Optional[Dict] = None
            for i in range(0, B, batch_size):
                wd_batch = Collate()(wd_list[i:i+batch_size])
                
                # run models
                for checkpoint in checkpoints:
                    model_ref = self.model_refs[checkpoint]
                    #print(f"Batch {(i//batch_size)+1}/{(B//batch_size)+1}: Running model: {model_ref}")
                    start_time = time.time()
                    wd_batch = self.models[checkpoint].infer(wd_batch)
                    print(f"{time.time() - start_time:.2f}s elapsed ({model_ref})")
                
                # move to CPU
                wd_batch = {k: v.cpu() if type(v) is torch.Tensor else v for k, v in wd_batch.items()}
                wd_batch = {k: v.float() if type(v) is torch.Tensor and v.dtype is torch.half else v for k, v in wd_batch.items()}
                
                # merge batch into output
                wd_out_batch = Collate().separate_batch(wd_batch)
                if wd_out is None:
                    wd_out = {k: [*wd_out_batch[k]] for k in wd_out_batch.keys()}
                else:
                    wd_out = {k: [*wd_out[k], *wd_out_batch[k]] for k in wd_out_batch.keys()}
            wd = {k: Collate().collate_left(v, k, same_n_channels='alignments' not in k) for k, v in wd_out.items()}
        
        return wd
    
    def infer_t2s(self, wd, checkpoints, remove_unneeded_models=False):
       return 

def get_highest_iter_checkpoint(directory):
    if os.path.exists(os.path.join(directory, 'best_validation_model')):
        return os.path.join(directory, 'best_validation_model')
    if os.path.exists(os.path.join(directory, 'latest_val_model')):
        return os.path.join(directory, 'latest_val_model')
    
    checkpoints = glob(os.path.join(directory, 'checkpoint_*'))
    checkpoints = [c for c in checkpoints if not c.endswith('_opt.pt')]
    if len(checkpoints):
        # return highest iter checkpoinst
        return next(reversed(sorted(checkpoints, key=lambda x: int(x.split("checkpoint_")[-1]))))

def run(audiopath, text, spkrname, dataloader, models, output_directory):
    wd = dataloader.get_od(audiopath, text, spkrname, arpa=False) # process inputs into text_ids, spkr_ids and so on
    print(wd.keys())
    for model in models:
        # run each model in order, updating the working dictionary as it goes
        wd = model.infer(wd)
        print(wd.keys())
    
    assert 'pr_wav' in wd
    # assert pr_wav exists and write to specified output directory
    for i in range(wd['pr_wav'].shape[0]):
        save_path = os.path.join(output_directory, f'{spkrname}_{i}.wav')
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        write(save_path, models[-1].h['audio_config']['sr'], wd['pr_wav'][i].view(-1).cpu().numpy())
    return wd

if __name__ == '__main__':
    # test worker
    
    # init worker
    weights_directory = "I:\\csruns"
    worker = Worker(weights_directory)
    
    # test worker.infer()
    n_b = 4
    name_i_offset = 0
    wd = {
        'audiopath': n_b*["L:\\TTS\\HiFiDatasets\\ClipperDataset\\Special source\\s2e14\\00_10_23_Applejack_Neutral__I thought cherries would be a Nice change from apples so, i Took the job and came here. That's it end of story.flac"],
        'text_raw' : n_b*["I thought cherries would be a Nice change from apples so, i Took the job and came here. That's it end of story."],
        'spkrname' : n_b*['Twilight'],
    }
    wd = {
#       'audiopath': n_b*["L:\\TTS\\HiFiDatasets\\Blizzard2011__Nancy\\NAN_447.flac"],
        'text_raw' : n_b*["Many users commented on the effectiveness of the new technology in promoting closer relationships among providers."],
        'spkrname' : n_b*['Nancy'],
    }
    checkpoints = [
#       "I:\\csruns\\text_to_prosody\\MAEPP\\outdir_15_Clipper_09_1024D_3K_4x4x2x2B_2L_0.0DO_0MHAtt\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\text_to_prosody\\DPMPP\\outdir_15_Clipper_09_1024D_3K_4x4x2x2B_2L_0.0DO_0MHAtt\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\text_to_mel\\FS2\\outdir_03_Clipper_ASR_based_durs\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\prosody_to_frp\\DPMFRPD\\outdir_15_Clipper\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\frp_to_mel\\DPMMELD\\outdir_16_Clipper-LNorm\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\text_to_mel\\tacotron2\\outdir_01_Pandora_CTCAlign\\weights\\best_cross_val.ptw",
        "I:\\csruns\\text_to_mel\\tacotron2\\outdir_01_Nancy_LSA_trial3\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\text_to_mel\\VDVAETTS\\outdir_01_Nancy_Tacotron2_based_durs_trial4_withPostNet\\weights\\best_cross_val.ptw",
#       "I:\\csruns\\text_to_mel\\FS2_DPM\\outdir_04_Clipper_Tacotron2_based_durs_trial1_MAE\\weights\\best_cross_val.ptw",
        "I:\\csruns\\vocoder\\FreGAN\\outdir_16_Pandora_with_SpkrEmbed\\weights\\best_cross_val.ptw",
#       "I:\\TTS\\CookieSpeech_checkpoints\\weights\\FreGAN\\testing_GAN_losses_03_1024init_C_spkrDis\\best_validation_model",
    ]
    wd = worker.infer(wd, checkpoints, unload_unneeded_models=False, b_arpabet=True)
    print(wd.keys())
    assert 'pr_wav' in wd
    assert 'wav_lens' in wd
    
    # assert pr_wav exists and write to specified output directory
    output_directory = os.path.join(__file__, '../../../infer_wavs')
    for i in range(wd['pr_wav'].shape[0]):
        save_path = os.path.join(output_directory, f'{wd["spkrname"][i]}_{i+name_i_offset}.wav')
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['pr_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
        print(save_path)
    
    
    if 0:
        wdi = {
            'audiopath': ["L:\\TTS\\HiFiDatasets\\Blizzard2011__Nancy\\NAN_447.flac"],
            'text_raw': ['Many users commented on the effectiveness of the new technology in promoting closer relationships among providers.'],
            'spkrname': ['Nancy'],
        }
        checkpoints = ["I:\\csruns\\vocoder\\FreGAN\\outdir_11_Pandora\\weights\\best_cross_val.ptw", ]
        wd = worker.infer(wdi, checkpoints, unload_unneeded_models=False, b_arpabet=False, extra_dl_feats={'gt_wav', 'wav_lens'})
        output_directory = os.path.join(__file__, '../../../infer_wavs')
        for i in range(wd['pr_wav'].shape[0]):
            save_path = os.path.join(output_directory, f'{wd["spkrname"][i]}_{i}_hifigan11P.wav')
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['pr_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
        for i in range(wd['gt_wav'].shape[0]):
            save_path = os.path.join(output_directory, f'{wd["spkrname"][i]}_{i}_gt.wav')
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['gt_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
        
        checkpoints = ["I:\\csruns\\vocoder\\FreGAN\\outdir_16_Pandora_without_SpkrEmbed_from_Scratch\\weights\\best_cross_val.ptw", ]
        wd = worker.infer(wdi, checkpoints, unload_unneeded_models=False, b_arpabet=False, extra_dl_feats={'gt_wav', 'wav_lens'})
        output_directory = os.path.join(__file__, '../../../infer_wavs')
        for i in range(wd['pr_wav'].shape[0]):
            save_path = os.path.join(output_directory, f'{wd["spkrname"][i]}_{i}_hifigan16P.wav')
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['pr_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
        
        checkpoints = ["I:\\csruns\\vocoder\\DDGANWave\\outdir_05_Nancy_NormalWaveGrad\\weights\\best_cross_val.ptw", ]
        wd = worker.infer(wdi, checkpoints, unload_unneeded_models=False, b_arpabet=False, extra_dl_feats={'gt_wav', 'wav_lens'})
        output_directory = os.path.join(__file__, '../../../infer_wavs')
        for i in range(wd['pr_wav'].shape[0]):
            save_path = os.path.join(output_directory, f'{wd["spkrname"][i]}_{i}_WaveGrad5N_100steps.wav')
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            write(save_path, worker.latest_h['dataloader_config']['audio_config']['sr'], wd['pr_wav'][i].view(-1)[:wd['wav_lens'][i]].cpu().numpy())
    
    print("Worker Test Completed")

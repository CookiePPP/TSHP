from typing import Tuple

import torch
from TSHP.utils.modules.core import ModelModule
from TSHP.modules.train_utils import guess_model_ref_from_path

def load_model(ref, path, h=None):
    # import the model's code
    name = "Model"
    modelmodule = getattr(
        __import__(f'CookieSpeech.models.{ref}.model', fromlist=[name]), name)
    # load checkpoint
    model, *_ = modelmodule.load_model(path, h=h, train=False)
    return model

def load_state_dict_from_path(path) -> Tuple[dict, dict]:
    d = torch.load(path, map_location='cpu')
    state_dict = d['state_dict']
    
    if 'iteration' not in state_dict:
        state_dict['iteration'] = torch.tensor(d['iteration'], dtype=torch.long)
    
    if 'secpr' not in state_dict:
        batch_size = d['h']['dataloader_config']['batch_size']
        average_audio_length = 3.952179378278209 # estimate average audio length using Pandora dataset as reference
        state_dict['secpr'] = torch.tensor(d['iteration'] * batch_size * average_audio_length, dtype=torch.double)
    
    if 'epoch' not in state_dict:
        average_epoch_size = 5137
        state_dict['epoch'] = torch.tensor(d['iteration'] // average_epoch_size, dtype=torch.double)
    
    if 'max_learning_rate' not in state_dict:
        state_dict['max_learning_rate'] = torch.tensor(0.0, dtype=torch.double)
    if 'best_cross_val' not in state_dict:
        state_dict['best_cross_val'] = torch.tensor(float('inf'), dtype=torch.double)
    if 'best_cross_val_secpr' not in state_dict:
        state_dict['best_cross_val_secpr'] = torch.tensor(0.0, dtype=torch.double)
    if 'lr_multiplier' not in state_dict:
        state_dict['lr_multiplier'] = torch.tensor(0.0, dtype=torch.double)
    
    return state_dict, d

def load_model_from_path(path: str) -> Tuple[ModelModule, dict]:
    state_dict, d = load_state_dict_from_path(path)
    
    # update hparams of old versions (CookieTTS / CookieTTS2 / CookieSpeech) to TSHP formatting
    h = d['h']
    
    if 'dataloader_config' not in h:
        h['dataloader_config'] = {}
    
    if 'audio_config' in h and 'audio_config' not in h['dataloader_config']:
        h['dataloader_config']['audio_config'] = h['audio_config']
    
    if 'stft_config' in h and 'stft_config' not in h['dataloader_config']:
        h['dataloader_config']['stft_config'] = h['stft_config']
    
    if 'pitch_config' in h and 'pitch_config' not in h['dataloader_config']:
        h['dataloader_config']['pitch_config'] = h['pitch_config']
    
    if 'text_config' in h and 'text_config' not in h['dataloader_config']:
        h['dataloader_config']['text_config'] = h['text_config']
    
    if 'bert_config' in h and 'bert_config' not in h['dataloader_config']:
        h['dataloader_config']['bert_config'] = h['bert_config']
    
    if 'n_symbols' not in h['model_config']:
        h['model_config']['n_symbols'] = h['dataloader_config']['text_config']['n_symbols']
    
    # get model class
    model_ref = guess_model_ref_from_path(path)
    
    # init object
    model = load_model(model_ref, path, h)
    model.load_state_dict(state_dict, strict=False)
    if 'spkrlist' in d:
        model.spkrlist = d['spkrlist']
    if 'textlist' in d:
        model.textlist = d['textlist']
    model.eval()
    
    # done
    return model, h


if __name__ == '__main__':
    # test load_state_dict
    paths = [
        "I:\\csruns\\text_to_prosody\\DPMPP\\outdir_11_Clipper\\weights\\best_cross_val.ptw",
        "I:\\csruns\\prosody_to_frp\\DPMFRPD\\outdir_11_Clipper\\weights\\best_cross_val.ptw",
        "I:\\csruns\\text_to_mel\\tacotron2\\outdir_01_Pandora_CTCAlign\\weights\\best_cross_val.ptw",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\ncwn_asr\\outdir_04_g_and_p_separated\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\ncwn_spkr_SIMC\\outdir_06_Adam_sqr_losses\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\FreGAN\\testing_GAN_losses_03_1024init_C_spkrDis_SFGAN\\checkpoint_176500",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\tacotron2\\outdir_05_AlignTest23_Clipper_RF1_CTCLoss_Pandora\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\fastspeech2_dec\\outdir_06_Pandora\\latest_val_model",
    ]
    for path in paths:
        state_dict = load_state_dict_from_path(path)[0]
        assert state_dict['iteration']  != 0.0
        assert state_dict['secpr'] != 0.0
        assert state_dict['epoch'] != 0.0
        assert state_dict['max_learning_rate'] is not None
        assert state_dict['best_cross_val'] is not None
        assert state_dict['best_cross_val_secpr'] is not None
        assert state_dict['lr_multiplier'] is not None
        
        assert state_dict['iteration'].dtype == torch.long
        assert state_dict['secpr'].dtype == torch.double
        assert state_dict['epoch'].dtype == torch.double
        assert state_dict['max_learning_rate'].dtype == torch.double
        assert state_dict['best_cross_val'].dtype == torch.double
        assert state_dict['best_cross_val_secpr'].dtype == torch.double
        assert state_dict['lr_multiplier'].dtype == torch.double
    print('Finished load_state_dict_from_path() unit test')
    
    # test load_model
    paths = [
        "I:\\csruns\\text_to_prosody\\DPMPP\\outdir_11_Clipper\\weights\\best_cross_val.ptw",
        "I:\\csruns\\prosody_to_frp\\DPMFRPD\\outdir_11_Clipper\\weights\\best_cross_val.ptw",
        "I:\\csruns\\text_to_mel\\tacotron2\\outdir_01_Pandora_CTCAlign\\weights\\best_cross_val.ptw",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\ncwn_asr\\outdir_04_g_and_p_separated\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\ncwn_spkr_SIMC\\outdir_06_Adam_sqr_losses\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\FreGAN\\testing_GAN_losses_03_1024init_C_spkrDis_SFGAN\\checkpoint_176500",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\tacotron2\\outdir_05_AlignTest23_Clipper_RF1_CTCLoss_Pandora\\best_inference_model",
        "I:\\TTS\\CookieSpeech_checkpoints\\weights\\fastspeech2_dec\\outdir_06_Pandora\\latest_val_model",
    ]
    for path in paths:
        model, h = load_model_from_path(path)
        assert 'dataloader_config' in h
        if True:
            dh = h['dataloader_config']
            assert 'audio_config' in dh
            assert 'stft_config' in dh
            assert 'pitch_config' in dh
            assert 'text_config' in dh
            assert 'bert_config' in dh
        assert hasattr(model, 'spkrlist')
    print('Finished load_model_from_path() unit test')
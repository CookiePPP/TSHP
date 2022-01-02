# imports
import logging
import warnings
from copy import deepcopy
from math import ceil
from traceback import print_exc
from typing import Optional, List, Dict

import syllables
import torch
import random
import os

from TSHP.utils.modules.utils import maybe_cat
from torch import Tensor

from TSHP.utils.dataset.spkr_csim.model import SpeakerEncoder
from TSHP.utils.modules.viterbi import viterbi
from TSHP.utils.saving.utils import safe_write

from TSHP.utils.dataset.metadata import get_dataset_meta
from TSHP.utils.dataset.audio.io import get_audio_from_path
from tqdm import tqdm
import TSHP.utils.warnings as w

# import dataset modules
from TSHP.utils.dataset.audio.audio_aug import AudioBandpass, AudioTrim, AudioLUFSNorm, AudioAug

from TSHP.utils.dataset.audio.stft import STFTModule
from TSHP.utils.dataset.audio.pitch import PitchModule

from TSHP.utils.dataset.text.processor import TextProcessor
# from TSHP.utils.dataset.moji.moji import BERT_wrapper
from TSHP.utils.dataset.moji.moji import TorchMoji
from TSHP.utils.dataset.BERT.bert_cpu_wrapper import BERT_wrapper


def weighted_choice(choices, seed=None): # https://stackoverflow.com/a/3679747
    """
    @param choices: list[Tuple[any, float]]
        list of [[value, prob], [value, prob], [value, prob], ...] 
    @param seed: Union[float, int, str]
        seed for random sampler
    
    @return: any
        random value from choices
    """
    total = sum(w for c, w in choices)
    random_obj = random.Random(seed) if seed is not None else random
    r = random_obj.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


# https://stackoverflow.com/a/43116588
def latest_modified_date_filtered(directory, exts=None):
    if exts is None:
        exts = ['.wav', '.flac', '.ogg', '.txt', '.csv']

    def filepaths(directory):
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                yield os.path.join(root, filename)
    
    files = (filepath for filepath in filepaths(directory) if any(filepath.lower().endswith(x) for x in exts))
    latest = max(files, key=os.path.getmtime)
    return os.stat(latest).st_mtime


class filelist:
    def __init__(self, filelist_config):
        self.audio_filters = filelist_config['audio_filters']
        self.audio_rejects = filelist_config['audio_rejects']
        self.min_sr = filelist_config['min_sr']
        self.max_sr = filelist_config['max_sr']
        self.min_dur = filelist_config['min_dur']
        self.max_dur = filelist_config['max_dur']
        self.min_chars = filelist_config['min_chars']
        self.max_chars = filelist_config['max_chars']
        self.min_audiofiles_per_speaker = filelist_config['min_audiofiles_per_speaker']
        self.skip_empty_datasets = filelist_config['skip_empty_datasets']
        self.default_source = "default_source"
        self.default_source_type = "default_source_type"
    
    @staticmethod
    def split_filelist(dictlist, p_val, p_test, seed, custom_test_data):
        # split dictlist
        trainlist = []
        trainspeakers = set()
        vallist = []
        valspeakers = set()
        testlist = []
        testspeakers = set()
        for d in dictlist:
            speaker = d['speaker']
            if speaker not in trainspeakers:
                trainspeakers.add(speaker)
                trainlist.append(d)
            elif speaker not in valspeakers:
                valspeakers.add(speaker)
                vallist.append(d)
            elif speaker not in testspeakers:
                testspeakers.add(speaker)
                testlist.append(d)
            else:
                # randomly assign file to list (using seed + audiofile for determinism)
                p_train = 1.0 - p_val - p_test
                choice = weighted_choice([
                    ['train', p_train],
                    ['val'  , p_val  ],
                    ['test' , p_test ],
                ], seed=d['path'] + str(seed))
                if choice == 'train':
                    trainlist.append(d)
                elif choice == 'val':
                    vallist.append(d)
                elif choice == 'test':
                    testlist.append(d)
                else:
                    raise NotImplementedError
        
        # inf list is not expected to contain audio paths, only requirements are text
        inflist = custom_test_data
        
        return trainlist, vallist, testlist, inflist

    # old version
    # def get_spkrlist_from_dictlist(self, dictlist, spkrnames):
    #    spkrlist = []# [["dataset","name","id","source","source_type","duration"], ...]
    #    for spkr_id, spkrname in enumerate(spkrnames):
    #        clip = next(d for d in dictlist if d['speaker'] == spkrname)
    #        spkrlist.append((spkrname, spkr_id, clip['dataset'], clip['source'], clip['source_type'], spkrdurs[spkrname]/3600.))
    #    return spkrlist

    # new version
    def get_spkrlist_from_dictlist(self, dictlist, spkrdurs):
        speakers = set()  # speakers already done
        spkrlist = []  # [["dataset","name","id","source","source_type","duration"], ...]
        for d in dictlist:
            speaker = d['speaker']
            if speaker not in speakers:
                spkr_id = len(speakers)
                spkrlist.append(
                    (speaker, spkr_id, d['dataset'], d['source'], d['source_type'], spkrdurs[speaker] / 3600.)
                )
                speakers.add(speaker)
        return spkrlist
    
    def get_multidataset_filelist(self, datasets_dir):  # works for folders with many datasets inside, each speaker should have their own folder.
        """
        Used during anything related to the dataset. (e.g: training, evaluation, tuning, preprocessing, batch-inference)  \n
        Takes a dataset directory or directory of dataset directories  \n
        returns a list of dicts with each dict using the structure below;  \n
        {
            audio_file: filepath,
            transcript: str,
            voice: str,
            emotions: list[str],
            dataset: str,
            noise_level: str,
            source: str,
            source_type: str,
        }
        """
        w.print1(f'checking "{datasets_dir}" for datasets')
        
        metadirpath = os.path.join(datasets_dir, 'meta')
        os.makedirs(metadirpath, exist_ok=True)
        datasets = sorted(
            [x for x in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, x)) and 'meta' not in x])
        assert len(datasets), 'found 0 datasets in datasets_dir'
        dictlist = []
        for dataset in datasets:
            ds_dict = self(os.path.join(datasets_dir, dataset))
            if ds_dict is not None:
                dictlist.extend(ds_dict)
        
        spkrnames = sorted(list(set([d['speaker'] for d in dictlist])))
        w.print1(f"Found {len(dictlist)} Total Valid Files from {len(spkrnames)} Speakers "
            f'with {sum(d["dur"] for d in dictlist)/60.0:.2f} minutes of data '
            f'and {sum(len(d["quote"]) for d in dictlist):,.0f} text ids')
        
        spkrdurs = {k: 0. for k in spkrnames}
        for d in dictlist:
            spkrdurs[d['speaker']] += d['dur']
        w.print1(f"Average speaker duration is {(sum(spkrdurs.values()) / len(spkrdurs.keys())) / 60.0:.2f} minutes")
        
        spkrlist = self.get_spkrlist_from_dictlist(dictlist, spkrdurs)
        
        self.write_spkrlist_to_file(spkrlist, os.path.join(metadirpath, 'spkr_info.txt'))
        
        return dictlist, spkrlist
    
    def write_spkrlist_to_file(self, spkrlist, path):
        # find ideal width for each column
        max_chars = []
        for i in range(len(spkrlist[0])):
            max_chars.append(max([len(str(x[i])) for x in spkrlist]))

        # and write table to file
        with open(path, 'w') as f:
            for list in spkrlist:
                list = [f'{str(x).strip().ljust(max_chars[i])}' for i, x in enumerate(list)]
                list[1] = f'{list[1].strip().rjust(max_chars[1])}'
                f.write('|'.join(list) + '\n')

    def load_or_prompt_defaults(self, dataset_dir):
        dataset = os.path.split(dataset_dir)[-1]
        
        speaker_default = None
        speaker_default_path = os.path.join(dataset_dir, 'default_speaker.txt')
        if os.path.exists(speaker_default_path):
            with open(speaker_default_path, 'r', encoding='utf8') as f:
                speaker_default = f.read()
        if not speaker_default:
            speaker_default = dataset.split("__")[1] if "__" in dataset else dataset
            print(
                f'default speaker for "{dataset}" dataset is missing.\nPlease enter the name of the default speaker\nExamples: "Nancy", "Littlepip", "Steven"')
            print(f'Press Enter with no input to use "{speaker_default}"')
            usr_inp = input('> ')
            if len(usr_inp.strip()) != 0:
                speaker_default = usr_inp
            with open(speaker_default_path, 'w', encoding='utf8') as f:
                f.write(speaker_default)
            print("")
        
        source_default = None
        source_default_path = os.path.join(dataset_dir, 'default_source.txt')
        if os.path.exists(source_default_path):
            with open(source_default_path, 'r', encoding='utf8') as f:
                source_default = f.read()
        if not source_default:
            print(
                f'default source for "{dataset}" dataset is missing.\nPlease enter the default source\nExamples: "My Little Pony", "Team Fortress 2", "University of Edinburgh"')
            source_option = dataset.split("__")[0] if "__" in dataset else None
            if source_option is not None:
                source_default = source_option
                print(f'Press Enter with no input to use "{source_option}"')
            usr_inp = input('> ')
            if len(usr_inp.strip()) == 0 and source_option is None:
                usr_inp = self.default_source
                print(f'No input given, defaulting to "{self.default_source}"')
            if len(usr_inp.strip()) != 0:
                source_default = usr_inp.strip()
                self.default_source = source_default
            with open(source_default_path, 'w', encoding='utf8') as f:
                f.write(source_default)
            print("")
        
        source_type_default = None
        source_type_default_path = os.path.join(dataset_dir, 'default_source_type.txt')
        if os.path.exists(source_type_default_path):
            with open(source_type_default_path, 'r', encoding='utf8') as f: source_type_default = f.read()
        if not source_type_default:
            print(
                f'default source type for "{dataset}" dataset is missing.\nPlease enter the default source type\nExamples: "Show", "Audiobook", "Audiodrama", "Game", "Newspaper Extracts"')
            usr_inp = input('> ')
            if len(usr_inp.strip()) == 0:
                usr_inp = self.default_source_type
                print(f'No input given, defaulting to "{self.default_source_type}"')
            if len(usr_inp.strip()) > 0:
                source_type_default = usr_inp.strip()
                self.default_source_type = source_type_default
            with open(source_type_default_path, 'w', encoding='utf8') as f:
                f.write(source_type_default)
            print("")
        
        return speaker_default, source_default, source_type_default
    
    def get_dsmeta_config(self):
        config = [
            self.audio_filters,
            self.audio_rejects,
            self.skip_empty_datasets,
        ]
        return config

    def cache_result(self, dataset_dir, dictlist):
        # find latest last modified date (for just {'.wav','.flac','.ogg','.txt','.csv'} files)
        lastmodified = latest_modified_date_filtered(dataset_dir, exts=['.wav', '.flac', '.ogg', '.txt', '.csv'])
        
        # write result to disk
        config = self.get_dsmeta_config()
        safe_write(
            {'dictlist': dictlist, 'lastmodified': lastmodified, 'config': config},
            os.path.join(dataset_dir, 'dictlist_cache.pt'),
        )

    def maybe_load_cache(self, dataset_dir):
        cache_path = os.path.join(dataset_dir, 'dictlist_cache.pt')
        if not os.path.exists(cache_path):
            return None
        
        try:
            dict = torch.load(cache_path)
        except Exception as ex:
            print_exc()
            return None
        
        same_lastmodified = dict['lastmodified'] == latest_modified_date_filtered(dataset_dir, exts=['.wav', '.flac', '.ogg', '.txt', '.csv'])
        all_files_exist = all(os.path.exists(x['path']) for x in dict['dictlist'])
        same_config = self.get_dsmeta_config() == dict.get('config', {})
        if same_lastmodified and same_config and all_files_exist:
            return dict['dictlist']
        else:
            return None

    def __call__(self, dataset_dir, cache_result=True, return_none_on_empty=True):  # gives dictlist from directory.
        # This func is called on single datasets, not folders of many datasets
        # return cached result if exists and is valid
        w.print1(f'checking "{dataset_dir}" for audio files')
        dictlist = self.maybe_load_cache(dataset_dir)
        if dictlist is None:
            # create/process a new dictlist
            
            # get Speaker, Source and Source Type (either from text file or from User)
            default_speaker, default_source, default_source_type = self.load_or_prompt_defaults(dataset_dir)
            
            # find audiofiles + text and link them together
            dictlist = get_dataset_meta(dataset_dir,
                                        default_emotion='Neutral', default_noise_level='Clean',
                                        default_speaker=default_speaker, default_source=default_source,
                                        default_source_type=default_source_type,
                                        audio_ext=self.audio_filters, audio_rejects=self.audio_rejects,
                                        naming_system=None, skip_empty=self.skip_empty_datasets,
                                        )
            
            if dictlist is None and return_none_on_empty:
                return
            
            # get durations + sampling rates and filter out bad audio files
            new_dictlist = []
            for d in tqdm(dictlist):
                path = d['path']
                try:
                    audio, sr = get_audio_from_path(path, return_empty_on_exception=True)
                except Exception as ex:
                    w.print2(f'Failed to load "{path}"\n', ex, '\n')
                    continue
                d['dur'], d['sr'] = len(audio) / sr, sr
                new_dictlist.append(d)
            dictlist = new_dictlist
            
            if cache_result:
                self.cache_result(dataset_dir, dictlist)
        
        # filter out bad, short, long, low band, files
        new_dictlist = []
        for d in dictlist:
            quote = d['quote']
            if len(quote) < self.min_chars: continue
            if len(quote) > self.max_chars: continue
            
            if d['sr'] < self.min_sr: continue
            if d['sr'] > self.max_sr: continue
            if d['dur'] < self.min_dur: continue
            if d['dur'] > self.max_dur: continue
            new_dictlist.append(d)
        dictlist = new_dictlist
        w.print1(
            f'found {len(dictlist)} audio files to use '
            f'with {sum(d["dur"] for d in dictlist)/60.0:.2f} minutes of data '
            f'and {sum(len(d["quote"]) for d in dictlist):,.0f} text ids'
        )
        
        # filter out speakers without enough data
        spkr_n_files = {}
        for d in dictlist:  # get number of files for each speaker
            if d['speaker'] not in spkr_n_files:
                spkr_n_files[d['speaker']] = 1
            else:
                spkr_n_files[d['speaker']] += 1
            
            # then remove files from speakers with less than min_audiofiles
        valid_spkrs = set([k for k, v in spkr_n_files.items() if v >= self.min_audiofiles_per_speaker])
        dictlist = [d for d in dictlist if d['speaker'] in valid_spkrs]
        
        return dictlist


# dataloader class - does all the data loading, cannot be iterated on
class DataLoaderModule:
    def __init__(self,
                 args,
                 training: bool,
                 dataloader_config,
                 spkrlist,
                 allow_caching=True,
                 spkr_csim_path=None):
        self.args_raw = args
        self.args = set([k.replace('_same_spkr', '').replace('_diff_spkr', '') for k in args]) # unique args (with same_spkr and diff_spkr merged)
        self.training = training
        
        # extract module configs
        resampling_config = dataloader_config['resampling_config']
        audio_config = dataloader_config['audio_config']
        audio_data_augmentation_config = dataloader_config['audio_data_augmentation_config']
        stft_config = dataloader_config['stft_config']
        pitch_config = dataloader_config['pitch_config']
        text_config = dataloader_config['text_config']
        bert_config = dataloader_config['bert_config']
        
        # Misc Params
        self.sr = sr = audio_config['sr']
        self.hop_len = stft_config['hop_len']
        
        # Audio Processing
        self.audio_pass = AudioBandpass(**audio_config['band_pass_config'], sr=sr)
        self.audio_trim = AudioTrim(**audio_config['trim_config'])
        self.audio_norm = AudioLUFSNorm(sr, audio_config['target_lufs'])
        self.audio_aug = AudioAug(**audio_data_augmentation_config, sr=sr, )
        
        # Mel/Spectrogram Processing
        self.stft = STFTModule(sr=sr, **stft_config)
        
        # Pitch/Voiced Processing
        self.f0module = PitchModule(sr, stft_config['hop_len'], **pitch_config)
        
        # Text Processing
        self.p_arpabet = text_config.pop('p_arpabet')
        text_config['p_arpabet'] = 0.5
        self.text = TextProcessor(**text_config)
        if any(k in args for k in ('bert_embed',)):
            self.BERT = BERT_wrapper(**bert_config)
        if any(k in args for k in ('moji_embed',)):
            self.moji = TorchMoji()
        
        # Speaker IDs
        self.spkrlist = spkrlist
        self.spkrlookup = None # will become dict {spkrname: spkr_id, ...} if spkr_ids are needed
        
        # Speaker Encoder
        if any(k in args for k in ('csim_spkr_embed',)):
            self.spkrenc = SpeakerEncoder(spkr_csim_path)
        
        # Text Augmentation (not used for training)
        self._augment_transcript = False
        
        # Caching Params
        self.allow_caching = not bool(max(  # disable caching if doing data augmentation
            audio_data_augmentation_config['rescale_pitch'],
            audio_data_augmentation_config['rescale_speed'],
            audio_data_augmentation_config['rescale_volume'],
            audio_data_augmentation_config['add_whitenoise'],
            audio_data_augmentation_config['add_noise'],
        ) and allow_caching)
        self.text_config = [text_config, bert_config]
        self.stft_config = [audio_config, stft_config]
        self.f0_config   = [audio_config, pitch_config]
        
        # Misc
        self.mel_T_keys = {
            'gt_stft' , 'gt_mel',
            'gt_vol'  , 'gt_nrg',
            'gt_f0'   , 'gt_vo' ,
            'gt_sf0'  , 'gt_svo',
            'gt_logf0', 'gt_logsf0',
            'gt_ppg'
        }
        self.wav_T_keys = {'gt_wav', }
        self.txt_T_keys = {
              'bert_embed',   'text_ids',
            'g_bert_embed', 'g_text_ids',
            'p_bert_embed', 'p_text_ids'
        }
    
    def train(self, train=True):
        self.training = train
    
    def eval(self):
        self.train(False)
    
    def get_audio(self, wd):
        if 'gt_wav' in wd.keys(): return wd
        
        audio, sr = get_audio_from_path(wd['audiopath'], target_sr=self.sr)
        
        # denoise
        # NOT IMPLEMENTED
        
        # bandpass
        audio = self.audio_pass(audio)
        
        # trim
        audio = self.audio_trim(audio)
        
        # volume normalization
        audio = self.audio_norm(audio, wd['audiopath'])
        
        # augment
        if self.training:
            audio = self.audio_aug(audio)
        
        wd['gt_wav'] = audio[None, :, None]  # [1, wav_T, 1]
        wd['sec_lens'] = torch.tensor(len(audio)/sr).view(1, 1, 1)
        wd['sr'] = sr
        return wd
    
    def get_sec_lens(self, wd):
        if 'sec_lens' in wd: return wd
        wd = self.get_audio(wd)
        wd['sec_lens'] = torch.tensor(wd['gt_wav'].shape[1]/wd['sr'], dtype=torch.float).view(1, 1, 1)
        return wd
    
    def get_sr(self, wd):
        if 'sr' in wd.keys():
            return wd
        if self.sr is not None:
            wd['sr'] = self.sr
            return wd
        return self.get_audio(wd)
    
    def get_stft(self, wd, ext='_stft.pt'):
        if 'gt_stft' in wd.keys(): return wd
        
#       stftpath = wd['basepath'] + ext
#       do_cache = self.allow_caching and os.path.exists(stftpath)
#       if do_cache:
#           stft, stft_config = torch.load(stftpath, map_location='cpu')
#           if stft_config == self.stft_config:
#               wd['gt_stft'] = stft
#               return wd
        
        wd = self.get_audio(wd)
        wd['gt_stft'] = self.stft.from_audio_get_spect(wd['gt_wav'])  # [B, mel_T, n_stft]
#       if self.allow_caching:
#           safe_write([wd['gt_stft'], self.stft_config], stftpath)
        return wd
    
    def get_vol(self, wd, ext='_vol.pt'):
        if 'gt_vol' in wd.keys(): return wd
        
        volpath = wd['basepath'] + ext
        do_cache = self.allow_caching and os.path.exists(volpath)
        if do_cache:
            try:
                gt_vol, stft_config = torch.load(volpath, map_location='cpu')
                if stft_config == self.stft_config:
                    wd['gt_vol'] = gt_vol
                    return wd
            except Exception as ex:
                print_exc()
        
        wd = self.get_stft(wd)
        wd['gt_vol'] = self.stft.from_spect_get_perc_loudness(wd['gt_stft'])  # [B, mel_T, 1]
        if self.allow_caching:
            safe_write([wd['gt_vol'], self.stft_config], volpath)
        return wd
    
    def get_mel(self, wd, ext='_mel.pt'):
        if 'gt_mel' in wd.keys(): return wd
        
        melpath = wd['basepath'] + ext
        do_cache = self.allow_caching and os.path.exists(melpath)
        if do_cache:
            try:
                gt_mel, mel_config = torch.load(melpath, map_location='cpu')
                if mel_config == self.stft_config:
                    wd['gt_mel'] = gt_mel
                    return wd
            except Exception as ex:
                print_exc()
        
        wd = self.get_stft(wd)
        wd['gt_mel'] = self.stft.from_spect_get_mel(wd['gt_stft'])  # [B, mel_T, n_stft]
        if self.allow_caching:
            safe_write([wd['gt_mel'], self.stft_config], melpath)
        return wd
    
    def get_nrg(self, wd):
        if 'gt_nrg' in wd.keys(): return wd
        
        wd = self.get_mel(wd)
        wd['gt_nrg'] = wd['gt_mel'].sub(self.stft.log_clamp_val).pow(2).mean(dim=2).pow(0.5).unsqueeze(2)  # [B, mel_T, 1]
        return wd
    
    def get_f0(self, wd, ext='_f0.pt'):
        if 'gt_f0' in wd.keys(): return wd
        
        f0path = wd['basepath'] + ext
        do_cache = self.allow_caching and os.path.exists(f0path)
        if do_cache:
            gt_f0, gt_vo, f0_config = torch.load(f0path, map_location='cpu')
            wd = self.get_mel(wd)
            if f0_config == self.f0_config and wd['gt_mel'].shape[1]-1 <= gt_f0.shape[1] <= wd['gt_mel'].shape[1]+1:
                wd['gt_f0'] = gt_f0
                wd['gt_vo'] = gt_vo
                return wd
        
        wd = self.get_audio(wd)
        wd = self.get_mel(wd)
        gt_f0, gt_vo = self.f0module.get_pitch(wd['gt_wav'])
        mel_len = wd['gt_mel'].shape[1]
        wd['gt_f0'], wd['gt_vo'] = gt_f0[None, :mel_len, None], gt_vo[None, :mel_len, None]
        if self.allow_caching:
            safe_write([wd['gt_f0'], wd['gt_vo'], self.f0_config], f0path)
        return wd
    
    def get_vo(self, wd):
        return self.get_f0(wd)
    
    def get_sf0(self, wd, ext='_sf0.pt'):
        if 'gt_sf0' in wd.keys() and 'gt_svo' in wd.keys(): return wd
        
        f0path = wd['basepath'] + ext
        do_cache = self.allow_caching and os.path.exists(f0path)
        if do_cache:
            try:
                gt_sf0, gt_svo, f0_config = torch.load(f0path, map_location='cpu')
                wd = self.get_mel(wd)
                if f0_config == self.f0_config:
                    assert wd['gt_mel'].shape[1]-1 <= gt_sf0.shape[1] <= wd['gt_mel'].shape[1]+1
                    wd['gt_sf0'] = gt_sf0
                    wd['gt_svo'] = gt_svo
                    return wd
            except Exception as ex:
                print_exc()
        
        wd = self.get_audio(wd)
        wd = self.get_mel(wd)
        gt_sf0, gt_svo = self.f0module.get_softpitch(wd['gt_wav'])
        mel_len = wd['gt_mel'].shape[1]
        assert mel_len-1 <= gt_sf0.shape[0] <= mel_len+1, f'got mel_len of {mel_len} and gt_sfo len of {gt_sf0.shape[0]}'
        wd['gt_sf0'], wd['gt_svo'] = gt_sf0[None, :mel_len, None], gt_svo[None, :mel_len, None]#gt_sf0[None, :mel_len, None], gt_svo[None, :mel_len, None]
        if self.allow_caching:
            safe_write([wd['gt_sf0'], wd['gt_svo'], self.f0_config], f0path)
        return wd
    
    def get_svo(self, wd):
        return self.get_sf0(wd)

    def get_logf0(self, wd):
        if 'gt_logf0' in wd.keys(): return wd
        
        wd = self.get_f0(wd)
        wd['gt_logf0'] = self.f0module.get_logpitch_from_pitch(wd['gt_f0'])
        return wd

    def get_logsf0(self, wd):
        if 'gt_logsf0' in wd.keys(): return wd
        
        wd = self.get_sf0(wd)
        wd['gt_logsf0'] = self.f0module.get_logpitch_from_pitch(wd['gt_sf0'])
        return wd

    def get_ppg(self, wd, ext='_ppg.pt'):
        ppgpath = wd['basepath'] + ext
        assert os.path.exists(ppgpath), f'ppg file at "{ppgpath}" does not exist'
        wd['gt_ppg'] = torch.load(ppgpath, map_location='cpu').float()
        return wd

    def get_moji_embed(self, wd, ext='_gmoji.pt'):
        if 'moji_embed' in wd.keys(): return wd
        
        if 'basepath' in wd:
            mojipath = wd['basepath'] + ext
            do_cache = self.allow_caching and os.path.exists(mojipath)
            if do_cache:
                try:
                    moji_embed, text_config = torch.load(mojipath, map_location='cpu')
                    if text_config == self.text_config:
                        wd['moji_embed'] = moji_embed.view(1, 1, -1)
                        assert wd['moji_embed'].dim() >= 3
                        return wd
                except Exception as ex:
                    print_exc()
        
        wd = self.get_g_text(wd)
        wd['moji_embed'] = self.moji(wd['g_text']).view(1, 1, -1)
        
        if self.allow_caching and 'basepath' in wd:
            safe_write([wd['moji_embed'], self.text_config], mojipath)
        return wd
    
    def get_csim_spkr_embed(self, wd, ext='_spkract.pt'):
        if 'csim_spkr_embed' in wd.keys(): return wd
        
        if 'basepath' in wd:
            spkrcsimpath = wd['basepath'] + ext
            do_cache = self.allow_caching and os.path.exists(spkrcsimpath)
            if do_cache:
                try:
                    csim_spkr_embed, stft_config, model_identifier = torch.load(spkrcsimpath, map_location='cpu')
                    if stft_config == self.stft_config and model_identifier == self.spkrenc.model_identifier:
                        wd['csim_spkr_embed'] = csim_spkr_embed
                        return wd
                except Exception as ex:
                    print_exc()
        
        wd = self.get_mel(wd)
        wd['csim_spkr_embed'] = self.spkrenc.forward(wd['gt_mel'], None)
        
        if self.allow_caching and 'basepath' in wd:
            safe_write([wd['csim_spkr_embed'], self.stft_config, self.spkrenc.model_identifier], spkrcsimpath)
        return wd
    
    def get_g_bert_embed(self, wd, ext='_gbert.pt'):
        if 'g_bert_embed' in wd.keys(): return wd
        
        repeat_interleave = self.text_config[0].get('repeat_interleave', 1)
        if 'basepath' in wd:
            bertpath = wd['basepath'] + ext
            do_cache = self.allow_caching and os.path.exists(bertpath)
            if do_cache:
                try:
                    g_bert_embed, g_text, text_config = torch.load(bertpath, map_location='cpu')
                    if text_config == self.text_config:
                        wd['g_bert_embed'] = g_bert_embed.repeat_interleave(repeat_interleave, dim=1)
                        wd['g_text'] = g_text
                        return wd
                except Exception as ex:
                    print_exc()
        
        with torch.no_grad():
            wd = self.get_g_text(wd)
            wd['g_bert_embed'], wd['g_text'] = self.BERT(wd['g_text'])
        if self.allow_caching and 'basepath' in wd:
            safe_write([wd['g_bert_embed'], wd['g_text'], self.text_config], bertpath)
        wd['g_bert_embed'] = wd['g_bert_embed'].repeat_interleave(repeat_interleave, dim=1)
        return wd
    
    def get_g_text(self, wd):
        if 'g_text' in wd.keys(): return wd
        
        wd['g_text'] = self.text.get_cleaned_text(wd['text_raw'])
        return wd
    
    def get_g_text_ids(self, wd):
        if 'g_text_ids' in wd.keys(): return wd
        
        wd = self.get_g_text(wd)
        wd['g_text_ids'], wd['g_text_symbols'] = self.text.get_text_ids(wd['g_text'])
        return wd
    
    def get_sylps(self, wd):
        try:
            wav_T = next(v.shape[1] for k, v in wd.items() if k in self.wav_T_keys)
        except StopIteration:
            try:
                mel_T = next(v.shape[1] for k, v in wd.items() if k in self.mel_T_keys)
                wav_T = mel_T * self.hop_len
            except StopIteration:
                wav_T = self.get_audio(wd)['gt_wav'].shape[1]
        dur_s = wav_T / self.sr
        wd['gt_sylps'] = syllables.estimate(wd['g_text']) / dur_s
        return wd
    
    def get_p_bert_embed(self, wd):
        if 'p_bert_embed' in wd.keys(): return wd
        
        repeat_interleave = self.text_config[0].get('repeat_interleave', 1)
        wd = self.get_g_text(wd)
        wd = self.get_g_bert_embed(wd)
        wd['p_text'], wd['p_bert_embed'] = self.text.convert_to_phones_with_hdn(wd['g_text'], wd['g_bert_embed'][:, ::repeat_interleave])
        wd['p_bert_embed'] = wd['p_bert_embed'].repeat_interleave(repeat_interleave, dim=1)
        return wd
    
    def get_p_text(self, wd):
        if 'p_text' in wd.keys(): return wd
        
        wd = self.get_g_text(wd)
        wd['p_text'] = self.text.convert_to_phones(wd['g_text'])
        return wd
    
    def get_p_text_ids(self, wd):
        if 'p_text_ids' in wd.keys(): return wd
        
        wd = self.get_p_text(wd)
        wd['p_text_ids'], wd['p_text_symbols'] = self.text.get_text_ids(wd['p_text'])
        return wd
    
    def get_phops(self, wd):
        try:
            wav_T = next(v.shape[1] for k, v in wd.items() if k in self.wav_T_keys)
        except StopIteration:
            try:
                mel_T = next(v.shape[1] for k, v in wd.items() if k in self.mel_T_keys)
                wav_T = mel_T * self.hop_len
            except StopIteration:
                wav_T = self.get_audio(wd)['gt_wav'].shape[1]
        dur_s = wav_T / self.sr
        wd['gt_phops'] = len(self.get_p_text(wd)['p_text']) / dur_s
        return wd
    
    def get_spkr_ids(self, wd):
        if 'spkr_ids' in wd.keys(): return wd
        assert hasattr(self, 'spkrlist') and self.spkrlist is not None, "data loader requires spkrlist to get spkr_ids"
        if not hasattr(self, 'spkrlookup') or self.spkrlookup is None:
            self.spkrlookup = {spkrname: spkr_id for spkrname, spkr_id, *_ in self.spkrlist}
        wd['spkr_ids'] = torch.tensor(self.spkrlookup[wd['spkrname']]).long().view(-1, 1, 1)
        return wd
    
    def get_alignments(self, wd):
        if 'alignments' in wd.keys():
            return wd
        if 'g_alignments' in wd.keys() and 'p_alignments' in wd.keys():
            return wd
        # load g + p alignments from disk
        
        alignpath = wd['basepath'] + '_align.pt'
        wd['g_alignments'] = torch.load(alignpath, map_location='cpu')# [1, txt_T, mel_T]
        if 'g_text_ids' in wd:
            if wd['g_alignments'].shape[1]%wd['g_text_ids'].shape[1] == 0:
                wd['g_alignments'] = wd['g_alignments'][:, ::wd['g_alignments'].shape[1]//wd['g_text_ids'].shape[1]]
            assert wd['g_text_ids'].shape[1] == wd['g_alignments'].shape[1], f"got {wd['g_text_ids'].shape[1]} and {wd['g_alignments'].shape[1]} for text lens and alignment lens respectively"
        
        alignpath = wd['basepath'] + '_plign.pt'
        wd['p_alignments'] = torch.load(alignpath, map_location='cpu')# [1, txt_T, mel_T]
        if 'p_text_ids' in wd:
            if wd['p_alignments'].shape[1]%wd['p_text_ids'].shape[1] == 0:
                wd['p_alignments'] = wd['p_alignments'][:, ::wd['p_alignments'].shape[1]//wd['p_text_ids'].shape[1]]
            assert wd['p_text_ids'].shape[1] == wd['p_alignments'].shape[1], f"got {wd['p_text_ids'].shape[1]} and {wd['p_alignments'].shape[1]} for text lens and alignment lens respectively"
        return wd
    
    def get_hard_alignments(self, wd, fast=False):
        if 'hard_alignments' in wd.keys():
            return wd
        if 'g_hard_alignments' in wd.keys() and 'p_hard_alignments' in wd.keys():
            return wd
        wd = self.get_alignments(wd)
        
        if fast:
            wd['p_hard_alignments'] = wd["p_alignments"].mul(0.0).float().scatter(1, wd["p_alignments"].argmax(1, True), 1.0)
            wd['g_hard_alignments'] = wd["g_alignments"].mul(0.0).float().scatter(1, wd["g_alignments"].argmax(1, True), 1.0)
        else:
            alignments = maybe_cat([wd["p_alignments"].float().clamp(1e-8).log(), wd["g_alignments"].float().clamp(1e-8).log()], dim=0, pad_to_common_lengths=True)
            hard_alignments = viterbi(
                alignments,
                enc_lens=torch.tensor([wd["p_alignments"].shape[1], wd["g_alignments"].shape[1]]).view(-1),
                dec_lens=torch.tensor([wd["p_alignments"].shape[2], wd["g_alignments"].shape[2]]).view(-1),
                n_rows_forward=3,
            )# [1, txt_T, mel_T]
            wd['p_hard_alignments'], wd['g_hard_alignments'] = hard_alignments.chunk(2, dim=0)
            wd['p_hard_alignments'] = wd['p_hard_alignments'][:, :wd["p_alignments"].shape[1]]
            wd['g_hard_alignments'] = wd['g_hard_alignments'][:, :wd["g_alignments"].shape[1]]
        
        assert wd['p_hard_alignments'].shape == wd["p_alignments"].shape, f'got mismatching shapes {wd["p_hard_alignments"].shape} and {wd["p_alignments"].shape}'
        assert wd['g_hard_alignments'].shape == wd["g_alignments"].shape, f'got mismatching shapes {wd["g_hard_alignments"].shape} and {wd["g_alignments"].shape}'
        return wd
    
    def get_gt_sdur(self, wd):
        if 'gt_sdur' in wd.keys():
            return wd
        if 'g_gt_sdur' in wd.keys() and 'p_gt_sdur' in wd.keys():
            return wd
        wd = self.get_alignments(wd)
        for app in ['p_', 'g_']:
            wd[app+'gt_sdur'] = wd[app+'alignments'].sum(dim=2, keepdim=True)# [1, txt_T, mel_T]
        return wd
    
    def get_gt_dur(self, wd):
        if 'gt_dur' in wd.keys():
            return wd
        if 'g_gt_dur' in wd.keys() and 'p_gt_dur' in wd.keys():
            return wd
        wd = self.get_hard_alignments(wd)
        for app in ['p_', 'g_']:
            wd[app+'gt_dur'] = wd[app+'hard_alignments'].sum(dim=2, keepdim=True)# [1, txt_T, mel_T]
        return wd
    
    def get_gt_logdur(self, wd):
        wd = self.get_gt_dur(wd)
        for app in ['p_', 'g_']:
            wd[app+'gt_logdur'] = wd[app+'gt_dur'].float().clamp(min=0.5).log()
        return wd
    
    def get_gt_lg1dur(self, wd):
        wd = self.get_gt_dur(wd)
        for app in ['p_', 'g_']:
            wd[app+'gt_lg1dur'] = wd[app+'gt_dur'].float().add(1.0).log()
        return wd
    
    def get_aligned_values(self, wd, alignment_key, value_key, out_key, ignore_zeroes=False):
        assert 'gt_' in value_key # TODO: remove this requirement
        if not any([arg in (
                  'gt_'+out_key.split('gt_')[-1],
                'g_gt_'+out_key.split('gt_')[-1],
                'p_gt_'+out_key.split('gt_')[-1],) for arg in self.args]):
            return wd
        if 'gt_c_'+value_key.split('gt_')[-1] in wd.keys():
            return wd
        if 'g_gt_c_'+value_key.split('gt_')[-1] in wd.keys() and 'p_gt_c_'+value_key.split('gt_')[-1] in wd.keys():
            return wd
        
        if value_key not in wd and hasattr(self, f'get_{value_key.split("gt_")[-1].split("pr_")[-1]}'):
            wd = getattr(self, f'get_{value_key.split("gt_")[-1].split("pr_")[-1]}')(wd)
        
        for app in ['g_', 'p_']:
            align = wd[app+alignment_key].float() # [1, txt_T, mel_T]
            feat = wd[value_key].float() # [1, mel_T, C]
            if ignore_zeroes:
                non_zeros = (feat.abs().sum(2, True)!=0.0).view(-1) # [1, mel_T, 1]
                feat  = feat [:, non_zeros, :]# [1, mel_T-zeros, 1]
                align = align[:, :, non_zeros]# [1, txt_T, mel_T-zeros]
            values = align @ feat # [1, txt_T, mel_T] @ [1, mel_T, C] -> [1, txt_T, C]
            values /= wd[app+alignment_key].sum(dim=2, keepdim=True).clamp(min=0.01)# /= [1, txt_T, 1]
            wd[app+out_key] = values
        return wd
    
    def get_mel_lens(self, wd):
        if 'mel_lens' in wd.keys(): return wd
        
        if any(k in self.mel_T_keys for k in wd.keys()):
            wd['mel_lens'] = next(v.shape[1] for k, v in wd.items() if k in self.mel_T_keys)
            wd['mel_lens'] = torch.tensor(wd['mel_lens']).long().view(-1, 1, 1)
            assert wd['mel_lens'].squeeze() > 0
        else:
            wd = self.get_stft(wd)
            return self.get_mel_lens(wd)
        return wd
    
    def get_txt_lens(self, wd):
        if 'txt_lens' in wd.keys(): return wd
        
        wd = self.get_g_text_ids(wd)
        wd = self.get_p_text_ids(wd)
        wd['g_txt_lens'] = wd['g_text_ids'].shape[1]
        wd['p_txt_lens'] = wd['p_text_ids'].shape[1]
        wd['g_txt_lens'] = torch.tensor(wd['g_txt_lens']).long().view(-1, 1, 1)
        wd['p_txt_lens'] = torch.tensor(wd['p_txt_lens']).long().view(-1, 1, 1)
        assert wd['g_txt_lens'].squeeze() > 0
        assert wd['p_txt_lens'].squeeze() > 0
        return wd
    
    def get_wav_lens(self, wd):
        if 'wav_lens' in wd.keys(): return wd
        
        if any(k in self.wav_T_keys for k in wd.keys()):
            wd['wav_lens'] = next(v.shape[1] for k, v in wd.items() if k in self.wav_T_keys) if 'mel_T' not in wd else self.hop_len * wd['mel_T']
            wd['wav_lens'] = torch.tensor(wd['wav_lens']).long().view(-1, 1, 1)
            assert wd['wav_lens'].squeeze() > 0
            return wd
        elif 'mel_lens' in wd:
            wd['wav_lens'] = self.hop_len * wd['mel_lens']
            assert wd['wav_lens'].squeeze() > 0
            return wd
        else:
            wd = self.get_audio(wd)
            return self.get_wav_lens(wd)
    
    def maybe_augment_text(self, wd):
        if self._augment_transcript:
            for _ in range(ceil(len(wd['text_raw'])/32)):# how many augments to make (each one will be either flipping, deleting or adding a single character)
                self._augment_raw_text(wd)
        return wd
    
    def _augment_raw_text(self, wd):
        x = random.random()
        if x < 0.33:
            # randomly flip character
            i1 = random.randint(0, len(wd['text_raw']) - 1)
            i2 = random.randint(0, len(wd['text_raw']) - 2)
            if i2 >= i1: i2 += 1
            i1t = wd['text_raw'][i1]
            i2t = wd['text_raw'][i2]
            wd['text_raw'][i2] = i1t
            wd['text_raw'][i1] = i2t
        elif 0.33 <= x < 0.66:
            # randomly add extra character
            i1 = random.randint(0, len(wd['text_raw']) - 1)  # index to add character to
            i2 = random.randint(0, len(wd['text_raw']) - 1)  # character to add
            wd['text_raw'] = wd['text_raw'][:i1] + [wd['text_raw'][i2], ] + wd['text_raw'][i1:]
        elif 0.66 <= x < 1.00:
            # randomly remove character
            i1 = random.randint(0, len(wd['text_raw']) - 1)  # index to remove character
            wd['text_raw'] = wd['text_raw'][:i1] + wd['text_raw'][i1 + 1:]
        else:
            raise Exception
    
    def get_wd(self, audiopath, text, spkrname, dur: Optional[float] = None):
        assert spkrname is not None
        assert text     is not None
        
        # init working dictionary
        wd = {
            'text_raw' : text,
            'spkrname' : spkrname,
        }
        if audiopath is not None:
            wd['audiopath'] = audiopath
            wd['basepath' ] = os.path.splitext(audiopath)[0]
        if dur is not None:
            wd['sec_lens'] = torch.tensor(dur, dtype=torch.float).view(1, 1, 1)
        
        wd = self.maybe_augment_text(wd)
        
        if audiopath is not None:
            if any([arg in ('gt_wav') for arg in self.args]):
                wd = self.get_audio(wd)
                assert 'gt_wav' in wd
            
            if any([arg in ('sec_lens') for arg in self.args]):
                wd = self.get_sec_lens(wd)
            
            if any([arg in ('gt_stft',) for arg in self.args]):
                wd = self.get_stft(wd)
                assert 'gt_stft' in wd
            
            if any([arg in ('gt_vol',) for arg in self.args]):
                wd = self.get_vol(wd)
            
            if any([arg in ('gt_mel',) for arg in self.args]):
                wd = self.get_mel(wd)
                assert 'gt_mel' in wd
            
            if any([arg in ('gt_nrg',) for arg in self.args]):
                wd = self.get_nrg(wd)
            
            if any([arg in ('gt_f0',) for arg in self.args]):
                wd = self.get_f0(wd)
            
            if any([arg in ('gt_vo',) for arg in self.args]):
                wd = self.get_vo(wd)
            
            if any([arg in ('gt_sf0',) for arg in self.args]):
                wd = self.get_sf0(wd)
            
            if any([arg in ('gt_svo',) for arg in self.args]):
                wd = self.get_svo(wd)
            
            if any([arg in ('gt_logf0',) for arg in self.args]):
                wd = self.get_logf0(wd)
            
            if any([arg in ('gt_logsf0',) for arg in self.args]):
                wd = self.get_logsf0(wd)
            
            if any([arg in ('gt_ppg',) for arg in self.args]):
                wd = self.get_ppg(wd)
        
        # global conds
        if any([arg in ('sr',) for arg in self.args]):
            wd = self.get_sr(wd)
        
        #       if any([arg in ('orig_sr',) for arg in self.args]):
        #           wd = self.get_orig_r(wd) # calc sample rate using spectrogram magnitudes
        
        if any([arg in ('moji_embed',) for arg in self.args]):
            wd = self.get_moji_embed(wd)
            assert wd['moji_embed'].dim() >= 3
        
        if any([arg in ('csim_spkr_embed',) for arg in self.args]):
            wd = self.get_csim_spkr_embed(wd)
        
        # grapheme related
        if any([arg in ('g_text', 'text') for arg in self.args]):
            wd = self.get_g_text(wd)
        
        if any([arg in ('g_text_ids', 'text_ids', 'g_text_symbols', 'text_symbols') for arg in self.args]):
            wd = self.get_g_text_ids(wd)
        
        if any([arg in ('g_bert_embed', 'bert_embed',) for arg in self.args]):
            wd = self.get_g_bert_embed(wd)
        
        if any([arg in ('gt_sylps',) for arg in self.args]):
            wd = self.get_sylps(wd)
        
        # phoneme related
        if any([arg in ('p_text', 'text') for arg in self.args]):
            wd = self.get_p_text(wd)
        
        if any([arg in ('p_text_ids', 'text_ids', 'p_text_symbols', 'text_symbols') for arg in self.args]):
            wd = self.get_p_text_ids(wd)
        
        if any([arg in ('p_bert_embed', 'bert_embed') for arg in self.args]):
            wd = self.get_p_bert_embed(wd)
        
        if any([arg in ('gt_phops',) for arg in self.args]):
            wd = self.get_phops(wd)
        
        # speaker related
        if any([arg in ('spkr_ids',) for arg in self.args]):
            wd = self.get_spkr_ids(wd)
        
        # alignments
        if audiopath is not None:
            if any([arg in ('alignments', 'g_alignments', 'p_alignments',) for arg in self.args]) \
                    or any(['gt_sc_' in arg for arg in self.args]):
                wd = self.get_alignments(wd)
            
            if any([arg in ('hard_alignments', 'g_hard_alignments', 'p_hard_alignments',) for arg in self.args]) \
                    or any(['gt_hc_' in arg for arg in self.args]):
                wd = self.get_hard_alignments(wd)
            
            if any([arg in ('gt_sdur', 'g_gt_sdur', 'p_gt_sdur',) for arg in self.args]):
                wd = self.get_gt_sdur(wd)
            
            if any([arg in ('gt_dur', 'g_gt_dur', 'p_gt_dur',) for arg in self.args]):
                wd = self.get_gt_dur(wd)
            
            if any([arg in ('gt_logdur', 'g_gt_logdur', 'p_gt_logdur',) for arg in self.args]):
                wd = self.get_gt_logdur(wd)
            
            if any([arg in ('gt_lg1dur', 'g_gt_lg1dur', 'p_gt_lg1dur',) for arg in self.args]):
                wd = self.get_gt_lg1dur(wd)
            
            # char-level features
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_vol'   , out_key='gt_sc_vol'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_nrg'   , out_key='gt_sc_nrg'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_f0'    , out_key='gt_sc_f0'    , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_vo'    , out_key='gt_sc_vo'    , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_sf0'   , out_key='gt_sc_sf0'   , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_svo'   , out_key='gt_sc_svo'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_logf0' , out_key='gt_sc_logf0' , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_logsf0', out_key='gt_sc_logsf0', ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key=     'alignments', value_key='gt_ppg'   , out_key='gt_sc_ppg'   , ignore_zeroes=False)
            
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_vol'   , out_key='gt_hc_vol'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_nrg'   , out_key='gt_hc_nrg'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_f0'    , out_key='gt_hc_f0'    , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_vo'    , out_key='gt_hc_vo'    , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_sf0'   , out_key='gt_hc_sf0'   , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_svo'   , out_key='gt_hc_svo'   , ignore_zeroes=False)
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_logf0' , out_key='gt_hc_logf0' , ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_logsf0', out_key='gt_hc_logsf0', ignore_zeroes=True )
            wd = self.get_aligned_values(wd, alignment_key='hard_alignments', value_key='gt_ppg'   , out_key='gt_hc_ppg'   , ignore_zeroes=False)
        
        # lengths related
        if audiopath is not None:
            if any([arg in ('mel_lens',) for arg in self.args]):
                wd = self.get_mel_lens(wd)
            
            if any([arg in ('wav_lens',) for arg in self.args]):
                wd = self.get_wav_lens(wd)
        
        if any([arg in ('txt_lens',) for arg in self.args]):
            wd = self.get_txt_lens(wd)
        
        # check for any non-finite data
        for k, v in wd.items():
            if type(v) is Tensor:
                assert torch.isfinite(v).all(), f'{k} has non-finite elements!'
        
        return wd
    
    def assign_text_type(self, wd, arpa=None):
        if arpa is None: # None for training, will randomly pick APRA with self.p_arpabet
            if self.p_arpabet is True or self.p_arpabet == 1:
                wd['arpa'] = True
            elif self.p_arpabet is False or self.p_arpabet == 0:
                wd['arpa'] = False
            else:
                wd['arpa'] = random.Random(wd['audiopath']).random() < self.p_arpabet if not self.training else random.random() < self.p_arpabet
        elif type(arpa) is bool: # True/False for inference
            wd['arpa'] = arpa
        elif type(arpa) is str and arpa == 'max':
            wd['arpa'] = wd['p_txt_lens'] > wd['g_txt_lens']
        else:
            raise NotImplementedError
        
        # e.g: {'g_text': 0, 'p_text': 1} -> {'text': 0}
        # where p/g is randomly picked for each dict
        for k in list(wd.keys()):
            if k.startswith('g_') and ('p_' + k.split('g_')[-1]) in wd.keys():
                bk = k.split('g_')[-1]
                wd[bk] = wd['p_' + bk] if wd['arpa'] else wd['g_' + bk]
        
        return wd
    
    def get_od(self, audiopath, text, spkr_id, arpa: Optional[bool] = None, dur: Optional[float] = None):
        # get all data that could be required (using self.args as a guide for what to load)
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        
        if 0:# True to use Profiler
            from line_profiler import LineProfiler
            lp = LineProfiler()
            lp_wrapper = lp(self.get_wd)
            wd = lp_wrapper(audiopath, text, spkr_id, dur=dur)
            lp.print_stats()
        else:
            wd = self.get_wd(audiopath, text, spkr_id, dur=dur)
        
        # random pick grapheme/phoneme mode
        wd = self.assign_text_type(wd, arpa=arpa)
        
        # check for bert length
        if 'bert_embed' in wd and 'text_ids' in wd:
            assert wd['bert_embed'].shape[1] == wd['text_ids'].shape[1], f"text: \"{wd['text']}\" has different length to expanded BERT embed.\nGot {wd['bert_embed'].shape[1]} and {wd['text_ids'].shape[1]} for bert and ids"
        
        # filter out any intermediate data not in self.args
        od = {k: v for k, v in wd.items() if k in self.args}  # select outputs from working dict
        
        torch.set_grad_enabled(prev)
        return od
    
    def __call__(self, *args, **kwargs):
        return self.get_od(*args, **kwargs)


# dataset class
_marker = object()
class DatasetModule(torch.utils.data.Dataset):
    """
    Used during anything related to the dataset. (e.g: training, evaluation, tuning, preprocessing, batch-inference)  \n
    Converts a list of [transcript, audiofile, speaker] into data the model(s) can more easily use (e.g: mel, spectrogram, phonemes, logpitch, energy)  \n
    1) ...
    """
    def __init__(self, parent, training: bool, segment_config: dict, dictlist: Optional[List[Dict]] = None):
        self.data_loader = parent
        self.training = training
        self.dictlist = dictlist
        
        self.segment_size    = segment_config['segment_size']
        self.random_segments = segment_config['random_segments']
        
        self.same_spkr_idxlist_dict = {}
    
    def train(self, train=True):
        self.training = train
        self.data_loader.train(train)
    
    def eval(self):
        self.train(False)
    
    def truncate_outputs(self, od):
        mel_T = mel_offset = None
        if any(k in self.data_loader.mel_T_keys for k in od.keys()):
            mel_T = next(v.shape[1] for k, v in od.items() if k in self.data_loader.mel_T_keys)
            mel_offset = 0
            max_start = max(mel_T-(self.segment_size+1), 0)
            if max_start > 0 and self.random_segments:
                mel_offset = random.Random(od['audiopath']).randint(0, max_start) if not self.training else random.randint( 0, max_start)
            od = {k: v[:, mel_offset:mel_offset + self.segment_size, :] if k in self.data_loader.mel_T_keys else v for k, v in od.items()}
        
        if any(k in self.data_loader.wav_T_keys for k in od.keys()):
            wav_T = next(v.shape[1] for k, v in od.items() if
                         k in self.data_loader.wav_T_keys) if mel_T is None else self.data_loader.hop_len * mel_T
            if mel_offset is not None:
                wav_offset = mel_offset * self.data_loader.hop_len
            else:
                wav_offset = 0
                max_start = max(wav_T - (self.segment_size * self.data_loader.hop_len), 0)
                if max_start > 0 and self.random_segments:
                    wav_offset = random.Random(od['audiopath']).randint(0, max_start) if not self.training else random.randint( 0, max_start)
            od = {k: v[:, wav_offset:wav_offset+(self.segment_size*self.data_loader.hop_len), :] if k in self.data_loader.wav_T_keys else v for k, v in od.items()}
        
        # txt_T = next(v.shape[1] for k,v in od.items() if k in txt_T_keys)
        #
        
        if 'mel_lens' in od.keys():
            od['mel_lens'] = od['mel_lens'].clamp(max=self.segment_size)
        
        if 'wav_lens' in od.keys():
            od['wav_lens'] = od['wav_lens'].clamp(max=self.segment_size*self.data_loader.hop_len)
        
        if 'sec_lens' in od.keys():
            od['sec_lens'] = od['sec_lens'].clamp(max=(self.segment_size*self.data_loader.hop_len)/self.data_loader.sr)
        
        return od
    
    def pick_file(self, list_idx):  # might be updated with resampling or randomization in future
        return self.dictlist[list_idx]
    
    def get_largest_od(self, key='dur'):
        assert key in self.dictlist[0], f'{key} not found in dictlist'
        
        # get longest file by duration
        dur_list = [len(d[key]) if hasattr(d[key], '__len__') else d[key] for d in self.dictlist]
        maxdur_index = dur_list.index(max(dur_list))
        largest_d = self.dictlist[maxdur_index]
        
        # and load the data
        od = self.data_loader.get_od(largest_d['path'], largest_d['quote'], largest_d['speaker'], arpa='max', dur=largest_d.get('dur', None))
        
        # segment / truncate output tensors
        od = self.truncate_outputs(od)
        
        # add "*_same_spkr" and "*_diff_spkr" args to od
        od = self.maybe_add_same_spkr(dict, od)
        od = self.maybe_add_diff_spkr(dict, od)
        
        return od
    
    def _construct_same_spkr_idxlist_dict(self):
        self.same_spkr_idxlist_dict = {}
        for i, d in enumerate(self.dictlist):
            speaker = d['speaker']
            if speaker in self.same_spkr_idxlist_dict:
                self.same_spkr_idxlist_dict[speaker].append(i)
            else:
                self.same_spkr_idxlist_dict[speaker] = [i]
        self.spkr_n_clips_dict = {speaker: len(self.same_spkr_idxlist_dict[speaker]) for speaker in self.same_spkr_idxlist_dict.keys()}
        self.n_clips_total = sum(self.spkr_n_clips_dict.values())
    
    def get_random_same_spkr(self, speaker, audiopath=None):
        if speaker not in self.same_spkr_idxlist_dict:
            self.same_spkr_idxlist_dict[speaker] = [i for i, d in enumerate(self.dictlist) if d['speaker'] == speaker]
        
        rand_obj = random if audiopath is None or self.training else random.Random(audiopath)
        index = rand_obj.randint(0, len(self.same_spkr_idxlist_dict[speaker])-1)
        return self.dictlist[self.same_spkr_idxlist_dict[speaker][index]]
    
    def get_random_diff_spkr(self, speaker, audiopath=None):
        if not getattr(self, 'spkr_n_clips_dict', None):
            self._construct_same_spkr_idxlist_dict()
        
        diff_speaker = weighted_choice(
            [(k, v) for k, v in self.spkr_n_clips_dict.items() if k != speaker],
            seed=audiopath
        )
        
        rand_obj = random if audiopath is None or self.training else random.Random(audiopath)
        index = rand_obj.randint(0, len(self.same_spkr_idxlist_dict[diff_speaker])-1)
        return self.dictlist[self.same_spkr_idxlist_dict[diff_speaker][index]]
    
    def maybe_add_same_spkr(self, dict, od):
        # add '*_same_spkr' args
        if any(arg.endswith('_same_spkr') for arg in self.data_loader.args_raw):
            # pick file (with index)   # equivalent to self.dictlist[list_idx] if not using resampling
            same_spkr_dict = self.get_random_same_spkr(dict['speaker'], audiopath=dict['path'])
            # load data
            same_spkr_od = self.data_loader.get_od(
                same_spkr_dict['path'],
                same_spkr_dict['quote'],
                same_spkr_dict['speaker'],
                dur=same_spkr_dict.get('dur', None)
            )
            # segment / truncate output tensors
            same_spkr_od = self.truncate_outputs(same_spkr_od)
        
            for k, v in same_spkr_od.items():
                if k + '_same_spkr' in self.data_loader.args_raw or type(v) is not torch.Tensor:
                    od[k + '_same_spkr'] = v
        return od
    
    def maybe_add_diff_spkr(self, dict, od):
        # add '*_diff_spkr' args
        if any(arg.endswith('_diff_spkr') for arg in self.data_loader.args_raw):
            # pick file (with index)   # equivalent to self.dictlist[list_idx] if not using resampling
            diff_spkr_dict = self.get_random_diff_spkr(dict['speaker'], audiopath=dict['path'])
            # load data
            diff_spkr_od = self.data_loader.get_od(
                diff_spkr_dict['path'],
                diff_spkr_dict['quote'],
                diff_spkr_dict['speaker'],
                dur=diff_spkr_dict.get('dur', None)
            )
            # segment / truncate output tensors
            diff_spkr_od = self.truncate_outputs(diff_spkr_od)
            
            for k, v in diff_spkr_od.items():
                if k + '_diff_spkr' in self.data_loader.args_raw or type(v) is not torch.Tensor:
                    od[k + '_diff_spkr'] = v
        return od
    
    def __len__(self):
        assert hasattr(self, 'dictlist'), 'called __len__ but dictlist is not registered'
        return len(self.dictlist)
    
    def __getitem__(self, list_idx):
        assert hasattr(self, 'dictlist'), 'called __getitem__ but dictlist is not registered'
        self.data_loader.train(self.training)  # ensure _parent is in the same mode (incase _parent is being shared among multiple objects)
        
        # add normal args
        # pick file (with index)   # equivalent to self.dictlist[list_idx] if not using resampling
        d = self.pick_file(list_idx)
        # load data
        od = self.data_loader.get_od(d['path'], d['quote'], d['speaker'], dur=d.get('dur', None))
        
        # segment / truncate output tensors
        od = self.truncate_outputs(od)
        
        # add "*_same_spkr" and "*_diff_spkr" args to od
        od = self.maybe_add_same_spkr(d, od)
        od = self.maybe_add_diff_spkr(d, od)
        
        return od


class Collate:
    def __init__(self):
        self.mel_T_keys = {'_stft', '_vol', '_mel', '_nrg', '_f0', '_vo', '_sf0', '_svo', '_logf0', '_logsf0', '_ppg'}
        self.wav_T_keys = {'_wav', }
        self.txt_T_keys = {'bert_embed', 'text_ids',}
    
    def separate_batch(self, batch, ignore_non_collatable=False):
        """Seperate collated tensors back to lists of tensors and remove padded space""" 
        possible_lens = {k: v for k, v in batch.items() if '_lens' in k}
        out = {}
        for k in batch.keys():
            if isinstance(batch[k], (list, tuple)):
                out[k] = batch[k]
                continue
            elif type(batch[k]) is not torch.Tensor:
                if not ignore_non_collatable:
                    out[k] = batch[k]
                continue
            elif ignore_non_collatable and batch[k].dim() < 3:
                continue
            lens = self.get_lengths(batch, k, possible_lens=possible_lens)
            lens2 = None
            if 'alignments' in k:
                batch[k] = batch[k].transpose(1, 2)
                lens2 = self.get_lengths(batch, k, possible_lens=possible_lens)
                batch[k] = batch[k].transpose(1, 2)
            out[k] = self.seperate(batch[k], k, lens=lens, lens2=lens2)
        
        return out
    
    def seperate(self, x, key, lens=None, lens2=None):
        if type(x) is tuple:
            return list(x)
        if type(x) is not Tensor:
            return x
        if x.dim() == 0:
            return x
        
        x_list = []
        for i, xi in enumerate(x.unbind(0)):
            xi_sliced = xi if lens is None else xi[:lens[i]]
            if lens2 is not None:
                xi_sliced = xi_sliced[:, :lens2[i]]
            x_list.append(xi_sliced.unsqueeze(0))
        return x_list
    
    def get_lengths(self, batch: dict, key, possible_lens: Optional[dict] = None):
        if type(batch[key]) is not Tensor:
            return None
        
        if possible_lens is None:
            possible_lens = {k: v for k, v in batch.items() if '_lens' in k}
        
        assert batch[key].dim() >= 3, f'key: {key} : has {batch[key].dim()} dims, expected 3 or more.'
        
        # try to guess with zero-padded area of tensor
        mask = batch[key].abs().sum(2, True).gt(0.0) # [B, T, C] -> [B, T, 1]
        batch_lens = mask.sum(1, True) # [B, T, 1] -> [B, 1, 1]
        for lens in possible_lens.values():
            if batch_lens.to(lens).eq(lens).all():
                return lens
        
        # try to guess correct lens from tensor length and lens.max()
        matching_lens = [[k, batch[key].shape[1] == lens.max().item()] for k, lens in possible_lens.items()]
        n_matching_lens = sum([x[1] for x in matching_lens])
        if n_matching_lens:
            return next(possible_lens[k] for k, does_match in matching_lens if does_match is True)
        
        # try to guess using self attributes
        if any(key in x for x in self.mel_T_keys):
            if batch[key].shape[1] == batch['pr_mel_lens'].max():
                return batch['pr_mel_lens']
            else:
                return batch['mel_lens']
        if any(key in x for x in self.wav_T_keys):
            if batch[key].shape[1] == batch['pr_wav_lens'].max():
                return batch['pr_wav_lens']
            else:
                return batch['wav_lens']
        if any(key in x for x in self.txt_T_keys):
            if batch[key].shape[1] == batch['pr_txt_lens'].max():
                return batch['pr_txt_lens']
            else:
                return batch['txt_lens']
        
        # give up
        return None
    
    def collate_left(self, x_batch: list, key, batch_size=None, lengths=None, same_n_channels=True):
        assert batch_size is None or len(x_batch) == batch_size  # check batch size matches length of x_batch
        assert len(x_batch) > 0
        
        if all(type(x) is torch.Tensor for x in x_batch):  # all is Tensor
            # check lengths match expected values
            if lengths is not None:
                for i, x in enumerate(x_batch):
                    assert x.shape[1] == lengths[i], f'item {i} in {key} batch has unexpected length. Got {x.shape[1]}, expected {lengths[i]}'
            
            # check n_channels matches
            if same_n_channels:
                assert x_batch[0].dim() >= 3, f'{key} has unexpected shape. got {x_batch[0].shape}'
                n_channels = x_batch[0].shape[2]
                for i, x in enumerate(x_batch):
                    assert x.dim() >= 3, f'{key} has unexpected shape. got {[x.shape for x in x_batch]}'
                    assert n_channels == x.shape[2], f'item {i} in {key} batch has unexpected n_channels. Got {x.shape[2]}({i}) and {n_channels}(0) in the same batch.'
            
            # check dtypes match
            dtype = x_batch[0].dtype
            for i, x in enumerate(x_batch):
                assert dtype == x.dtype, f'item {i} in {key} batch has unexpected dtype. Got {x.dtype}({i}) and {dtype}(0) in the same batch.'
            
            # create blank tensor and assign x items to left aligned rows
            max_len = max(lengths) if lengths is not None else max([x.shape[1] for x in x_batch])
            max_C = max(x.shape[2] for x in x_batch)
            x_out = torch.zeros(len(x_batch), max_len, max_C, device=x_batch[0].device, dtype=x_batch[0].dtype)
            for i, x in enumerate(x_batch):
                x_out[i:i + 1, :x.shape[1], :x.shape[2]] = x
            
            return x_out  # return output
        
        elif all(type(x) is not torch.Tensor for x in x_batch):  # all not Tensor
            # if not tensor, inputs should be treated as [[T, ...],] *B
            
            # check lengths match expected values
            if lengths is not None:
                for i, x in enumerate(x_batch):
                    assert len(x) == lengths[i], f'item {i} in {key} batch has unexpected length. Got {len(x)}, expected {lengths[i]}'
            
            # check dtypes match
            dtype = type(x_batch[0])
            for i, x in enumerate(x_batch):
                assert dtype == type(x), f'item {i} in {key} batch has unexpected type. Got {type(x)}({i}) and {dtype}(0) in the same batch.'
            
            # No padding needed because lists. Just return input after checks are done
            return x_batch
        else:
            raise TypeError
    
    def __call__(self, batch):
        """
        Collate's training batch from __getitem__ input features
        (this the merging all the features loaded from each audio file into the same tensors so the model can process together)
        PARAMS
            batch: [{"text": text[txt_T], "gt_mel": gt_mel[n_mel, mel_T]}, {"text": text, "gt_mel": gt_mel}, ...]
        ------
        RETURNS
            out: {"text": text_batch, "gt_mel": gt_mel_batch}
        """
        od_keys = list(set([k for od in batch for k, v in od.items()]))  # e.g: ['gt_mel','gt_wav','gt_vol', ...] 
        
        B = len(batch)
        
        out = {}
        # get lengths and give to collate method
        for k in od_keys:
            out[k] = self.collate_left([item[k] for item in batch], k, batch_size=B, same_n_channels=not 'alignments' in k)
        
        return out

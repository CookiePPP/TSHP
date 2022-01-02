from unidecode import unidecode
import torch
import torch.nn.functional as F
import pathlib
from os.path import split, join

class ARPA:
    def __init__(self, dict_path, punc="!?,.;:␤#~-_'\"()[]\n►><"):
        self.arpadict = self.load_arpadict(dict_path)
        self.punc = punc
    
    def load_arpadict(self, dict_path):
        if dict_path == 'DEFAULT':
            dict_path = split(split(split(pathlib.Path(__file__).parent.resolve())[0])[0])[0]# CookieTTS/
            dict_path = join(dict_path, 'dict', 'merged.dict.txt')
        
        # load dictionary as lookup table
        arpadict = {unidecode(line.split()[0]): unidecode(' '.join(line.split()[1:]).strip()) for line in open(dict_path, 'r')}
        return arpadict
    
    def get(self, text):
        """Convert block of text into ARPAbet."""
        out = []
        for word in text.split(" "):
            end_chars = ''; start_chars = ''
            while any(elem in word for elem in self.punc) and len(word) > 1:
                if word[-1] in self.punc:
                    end_chars = word[-1] + end_chars
                    word = word[:-1]
                elif word[0] in self.punc:
                    start_chars = start_chars + word[0]
                    word = word[1:]
                else:
                    break
            try:
                word = "{" + str(self.arpadict[word.upper()]) + "}"
            except KeyError:
                pass
            out.append((start_chars + (word or '') + end_chars).rstrip())
        return ' '.join(out)
    
    def get_hdn(self, text, hidden):# [txt_T], [1, txt_T, hdn]
        """Convert block of text and hidden_states into ARPAbet."""
        hidden = hidden.squeeze(0)
        assert len(text) == hidden.shape[0], f'text and hidden are not the same length, got {len(text)} and {hidden.shape[0]} respectively.'
        out = []
        hdn_out = []
        text_split = text.split(" ")
        len_text_split = len(text_split)
        for i, word in enumerate(text_split):
            word_hdn, hidden = hidden[:len(word)], hidden[len(word)+1:]# [T, hdn], [txt_T-T_cum, hdn]
            end_chars = ''; start_chars = ''
            while any(elem in word for elem in self.punc) and len(word) > 1:
                if word[-1] in self.punc:
                    end_chars = word[-1] + end_chars
                    word = word[:-1]
                elif word[0] in self.punc:
                    start_chars = start_chars + word[0]
                    word = word[1:]
                else:
                    break
            
            try:
                arpa_word = str(self.arpadict[word.upper()])
                word = "{"+arpa_word+"}"
                n_tokens = len(start_chars)+len(arpa_word.split(" "))+len(end_chars.rstrip())
                word_hdn = F.interpolate(
                    word_hdn.unsqueeze(0).transpose(1, 2), size=n_tokens
                ).squeeze(0).transpose(0, 1).unbind(0)# -> [hdn,]*T
            except KeyError:
                pass
            out.append((start_chars + (word or '') + end_chars).rstrip())
            hdn_out.extend(word_hdn)
            if i+1<len_text_split:
                hdn_out.append(word_hdn[0].new_zeros(word_hdn[0].shape))
        return ' '.join(out), torch.stack(hdn_out).unsqueeze(0)
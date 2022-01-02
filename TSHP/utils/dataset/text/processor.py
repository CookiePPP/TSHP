# imports
import torch
from unidecode import unidecode
from .numbers import normalize_numbers
from .ARPA import ARPA
from .sequence import SequenceModule

def force_lowercase(text):
    return text.lower()

def remove_camelcase(text, punctuation):# split on spaces, (for all non-punctuation chars, if not all upper, call lower())
    words = text.split(" ")
    out_words = []
    for word in words:
        if not len(word): continue
        before_word_punc = ''
        after_word_punc = ''
        while word[0] in punctuation:
            before_word_punc+=word[0]
            word = word[1:]
        while word[-1] in punctuation:
            before_word_punc = word[-1] + before_word_punc
            word = word[:-1]
        if word.upper() != word:
            word = word.lower()
        out_words.append(before_word_punc+word+after_word_punc)
    return ' '.join(out_words)

def expand_numbers(text):
    return normalize_numbers(text)

def force_ascii(text):
    return unidecode(text)

def collapse_whitespace(text):
    return text.strip()

def collapse_multispace(text):# https://stackoverflow.com/a/15913564
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text

def normalize_surprise(text):
    text = text.replace('?!', '!?')
    return text

def remove_end_semicolon(text, punctuation):
    if text[-1] == ';':
        text = text[:-1].strip()
        if text[-1] not in punctuation:
            text = text+'.'
    return text

def add_period_if_missing(text, punctuation):
    if text[-1] not in punctuation:
        text+='.'
    return text


class TextProcessor:
    def __init__(self,
            letters, punctuation, start_token, stop_token, remove_tokens, banned_tokens,
            force_lowercase, remove_camelcase, expand_numbers, force_ascii, collapse_whitespace, collapse_multispace, normalize_surprise, remove_end_semicolon, add_period_if_missing,
            p_arpabet, dict_path, arpabet_symbols, arpabet_style, repeat_interleave=1, n_symbols=None,
        ):
        # valid symbols to use
        self.letters     = letters
        self.punctuation = punctuation
        
        # append/prepend after cleaning
        self.start_token = start_token
        self.stop_token  = stop_token
        
        self.remove_tokens = remove_tokens
        self.banned_tokens = banned_tokens
        
        # text cleaning methods to use
        self.force_lowercase       = force_lowercase
        self.remove_camelcase      = remove_camelcase
        self.expand_numbers        = expand_numbers
        self.force_ascii           = force_ascii
        self.collapse_whitespace   = collapse_whitespace
        self.collapse_multispace   = collapse_multispace
        self.normalize_surprise    = normalize_surprise
        self.remove_end_semicolon  = remove_end_semicolon
        self.add_period_if_missing = add_period_if_missing
        
        # arpabet
        self.dict_path     = dict_path
        self.arpabet_style = arpabet_style# Choice['lookup','g2p', 'deepg2p','dataset_lookup']
        self.arpabet       = list(['@'+s for s in arpabet_symbols])
        self.p_arpabet     = p_arpabet
        self.arpa = ARPA(dict_path, self.punctuation)
        
        # text -> id module
        self.repeat_interleave = repeat_interleave
        self.seqmodule = SequenceModule(self.get_symbols())
        
        assert n_symbols is None or n_symbols == len(self.get_symbols()) 
        
    def get_symbols(self):
        return ['_', *self.letters, *self.punctuation, *self.arpabet]
    
    def get_cleaned_text(self, text):
        if self.force_ascii:
            text = force_ascii(text)
        if self.force_lowercase:
            text = force_lowercase(text)
        if self.remove_camelcase:
            text = remove_camelcase(text, self.punctuation)
        if self.expand_numbers:
            text = expand_numbers(text)
        if self.collapse_whitespace:
            text = collapse_whitespace(text)
        if self.collapse_multispace:
            text = collapse_multispace(text)
        if self.normalize_surprise:
            text = normalize_surprise(text)
        if self.remove_end_semicolon:
            text = remove_end_semicolon(text, self.punctuation)
        if self.add_period_if_missing:
            text = add_period_if_missing(text, self.punctuation)
        
        # add start/stop tokens (if used)
        text = str(self.start_token) + text + str(self.stop_token)
        
        # convert text to IDs then back to text. If anything is missing then it will be very obvious and easy to tell so you don't need to perform model surgery.
        text = self.seqmodule.sequence_to_text(self.seqmodule.text_to_sequence(text)[0])
        if self.collapse_multispace:
            text = collapse_multispace(text)
        return text
    
    def convert_to_phones(self, text):
        phones = self.arpa.get(text)
        return phones
    
    def convert_to_phones_with_hdn(self, text, hidden):# str[gtxt_T], Tensor[1, gtxt_T, hdn]
        phones, phidden = self.arpa.get_hdn(text, hidden)
        return phones, phidden# str[ptxt_T], Tensor[1, ptxt_T, hdn]
    
    def get_text_ids(self, text):
        text_ids, text_symbols = self.seqmodule.text_to_sequence(text)# [txt_T], str[txt_T]
        text_ids = torch.tensor(text_ids).long()[None, :, None]
        if self.repeat_interleave > 1:
            text_ids = text_ids.repeat_interleave(self.repeat_interleave, dim=1)
            text_symbols_out = []
            for sym in text_symbols:
                text_symbols_out.extend([sym, ]*self.repeat_interleave)
            text_symbols = text_symbols_out
            del text_symbols_out
        return text_ids, text_symbols# [1, txt_T, 1], str[txt_T]

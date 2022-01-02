""" from https://github.com/keithito/tacotron """
import re

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

class SequenceModule():
    def __init__(self, symbols):
        self.symbols = symbols
        
        # string -> symbols -> IDs
        
        # Mappings from symbol to numeric ID and vice versa:
        self.arpabet_set = set([x for x in self.symbols if x.startswith('@')])
        self._symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(symbols)}
    
    def text_to_sequence(self, text, cleaner_names=None):
        '''Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    
            The text can optionally have ARPAbet sequences enclosed in curly braces embedded
            in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."
    
            Args:
                text: string to convert to a sequence
                cleaner_names: names of the cleaner functions to run the text through
    
            Returns:
                List of integers corresponding to the symbols in the text
        '''
        sequence = []
        symbol_list = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                symbols = self._string_to_symbols(text)
                symbol_list.extend(symbols)
                sequence.extend(self._symbols_to_sequence(symbols))
                break
            symbols = self._string_to_symbols(m.group(1))
            symbol_list.extend(symbols)
            sequence.extend(self._symbols_to_sequence(symbols))
            
            symbols = self._arpastring_to_symbols(m.group(2))
            symbol_list.extend(symbols)
            sequence.extend(self._arpabet_to_sequence(symbols))
            
            text = m.group(3)
        return sequence, symbol_list
    
    def sequence_to_text(self, sequence):
        '''Converts a sequence of IDs back to a string'''
        result = ''
        for symbol_id in sequence:
            if symbol_id in self._id_to_symbol:
                s = self._id_to_symbol[symbol_id]
                # Enclose ARPAbet back in curly braces:
                if len(s) > 1 and s[0] == '@':
                    s = '{%s}' % s[1:]
                result += s
        return result.replace('}{', ' ')
    
    def _arpastring_to_symbols(self, string):# 'AH P L' -> ['AH', 'P', 'L']
        return [s for s in string.split() if '@'+s in self.arpabet_set]
    
    def _arpabet_to_sequence(self, symbols):
        return self._symbols_to_sequence(['@' + s for s in symbols])
    
    def _string_to_symbols(self, string):
        return [s for s in string if self._should_keep_symbol(s)]
    
    def _symbols_to_sequence(self, symbols):
        return [self._symbol_to_id[s] for s in symbols if self._should_keep_symbol(s)]
    
    def _should_keep_symbol(self, s):
        return s in self._symbol_to_id and s != '_'

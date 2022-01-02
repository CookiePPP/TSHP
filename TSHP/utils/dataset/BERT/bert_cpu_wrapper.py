import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import unicodedata
from unidecode import unidecode

# https://github.com/huggingface/transformers/blob/42fe0dc23e4a7495ebd08185f5850315a1a12dc0/src/transformers/tokenization_utils.py#L76-L88
def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class BERT_wrapper():
    def __init__(self, hdn_layer, bert_model, bert_embed_dim):
        self.hdn_layer = hdn_layer
        self.bert_embed_dim = bert_embed_dim
        
        # Load pre-trained model tokenizer (vocabulary)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=bool('uncased' in bert_model))
        
        # Load pre-trained model (weights)
        self.bert_model = BertModel.from_pretrained(bert_model).eval()
    
    def __call__(self, text, output_layer=11):
        text = unidecode(text)
        
        # Convert Text to Tokens
        tokenized_text = self.bert_tokenizer.tokenize("[CLS] "+text+" [SEP]")
        tokenized_text_str = ""
        
        # Convert tokens to ids
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_text)
        
        # Convert inputs to PyTorch tensors
        tokens_tensor    = torch.tensor([indexed_tokens])           # torch.Size([1, 14])
        segments_tensors = torch.tensor([0,]*tokens_tensor.shape[1])# torch.Size([14])
        
        with torch.no_grad():# Predict hidden states features for each layer
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)# We have a hidden states for each of the 12 layers in model bert-base-uncased
        hidden_out = encoded_layers[output_layer].squeeze(0)
        
        right_text = text.lower()
        expanded_hidden_out = []
        is_early_punc = True
        was_misc_punc = False
        for i, (token_hdn, token_str) in enumerate(zip(hidden_out[1:-1], tokenized_text[1:-1])):
            token_strn = token_str.replace("##", "")
            if right_text.count(token_strn):
                text_seg, right_text = right_text.split(token_strn, 1)
            else:
                text_seg = '####'
            is_incomplete_word = "##" in token_str
            expanded_hidden_out.extend(token_hdn.repeat(len(token_str.replace("##", "")), 1).unbind(0))
            tokenized_text_str+=token_str.replace("##", "")
            
            if right_text.startswith(" "):
                expanded_hidden_out.append(token_hdn.new_zeros(token_hdn.shape))
                tokenized_text_str+=" "
            
            is_early_punc = is_early_punc and all(_is_punctuation(t) for t in token_str)
            was_misc_punc = any(x in token_str for x in ("'","-"))
        expanded_hidden_out = torch.stack(expanded_hidden_out)
        
        assert expanded_hidden_out.shape[0] == len(text), f"length of text does not match BERT output length. Got text length of {len(text)} and BERT length of {expanded_hidden_out.shape[0]}\ntext:  {text}\ntoken: {tokenized_text_str}\n"
        assert expanded_hidden_out.shape[1] == self.bert_embed_dim, f"bert_embed_dim doesn't match output dim, got {self.bert_embed_dim} expected {expanded_hidden_out.shape[1]}"
        return expanded_hidden_out.unsqueeze(0), text# [1, txt_T, BERT_hdn], List[str]
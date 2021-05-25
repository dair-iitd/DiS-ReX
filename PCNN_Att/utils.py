from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
import unicodedata
import six

class MaxPool(nn.Module):

    def __init__(self, kernel_size, segment_num=None):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
        hidden_size: hidden size
        """
        super().__init__()
        self.segment_num = segment_num
        if self.segment_num != None:
            self.mask_embedding = nn.Embedding(segment_num + 1, segment_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros((1, segment_num)), np.identity(segment_num)], axis=0)))
            self.mask_embedding.weight.requires_grad = False
            self._minus = -100
        self.pool = nn.MaxPool1d(kernel_size)

    def forward(self, x, mask=None):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        if mask is None or self.segment_num is None or self.segment_num == 1:
            x = x.transpose(1, 2) # (B, L, I_EMBED) -> (B, I_EMBED, L)
            x = self.pool(x).squeeze(-1) # (B, I_EMBED, 1) -> (B, I_EMBED)
            return x
        else:
            B, L, I_EMBED = x.size()[:3]
            # mask = 1 - self.mask_embedding(mask).transpose(1, 2).unsqueeze(2) # (B, L) -> (B, L, S) -> (B, S, L) -> (B, S, 1, L)
            # x = x.transpose(1, 2).unsqueeze(1) # (B, L, I_EMBED) -> (B, I_EMBED, L) -> (B, 1, I_EMBED, L)
            # x = (x + self._minus * mask).contiguous().view([-1, I_EMBED, L]) # (B, S, I_EMBED, L) -> (B * S, I_EMBED, L)
            # x = self.pool(x).squeeze(-1) # (B * S, I_EMBED, 1) -> (B * S, I_EMBED)
            # x = x.view([B, -1])  # (B, S * I_EMBED)
            # return x
            mask = 1 - self.mask_embedding(mask).transpose(1, 2)
            x = x.transpose(1, 2)
            pool1 = self.pool(x + self._minus * mask[:, 0:1, :])
            pool2 = self.pool(x + self._minus * mask[:, 1:2, :])
            pool3 = self.pool(x + self._minus * mask[:, 2:3, :])

            x = torch.cat([pool1, pool2, pool3], 1)
            # x = x.squeeze(-1)
            return  x

class AvgPool(nn.Module):

    def __init__(self, kernel_size, segment_num=None):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
        hidden_size: hidden size
        """
        super().__init__()
        self.segment_num = segment_num
        if self.segment_num != None:
            self.mask_embedding = nn.Embedding(segment_num + 1, segment_num)
            self.mask_embedding.weight.data.copy_(torch.FloatTensor(np.concatenate([np.zeros(segment_num), np.identity(segment_num)], axis = 0)))
            self.mask_embedding.weight.requires_grad = False
        self.pool = nn.AvgPool1d(kernel_size)

    def forward(self, x, mask=None):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        if mask == None or self.segment_num == None or self.segment_num == 1:
            x = x.transpose(1, 2) # (B, L, I_EMBED) -> (B, I_EMBED, L)
            x = self.pool(x).squeeze(-1) # (B, I_EMBED, 1) -> (B, I_EMBED)
            return x
        else:
            B, L, I_EMBED = x.size()[:2]
            mask = self.mask_embedding(mask).transpose(1, 2).unsqueeze(2) # (B, L) -> (B, L, S) -> (B, S, L) -> (B, S, 1, L)
            x = x.transpose(1, 2).unsqueeze(1) # (B, L, I_EMBED) -> (B, I_EMBED, L) -> (B, 1, I_EMBED, L)
            x = (x * mask).view([-1, I_EMBED, L]) # (B, S, I_EMBED, L) -> (B * S, I_EMBED, L)
            x = self.pool(x).squeeze(-1) # (B * S, I_EMBED, 1) -> (B * S, I_EMBED)
            x = x.view([B, -1]) - self._minus # (B, S * I_EMBED)
            return x

class CNN(nn.Module):

    def __init__(self, input_size=50, hidden_size=256, dropout=0, kernel_size=3, padding=1, activation_function=F.relu):
        """
        Args:
            input_size: dimention of input embedding
            kernel_size: kernel_size for CNN
            padding: padding for CNN
            hidden_size: hidden size
        """
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=padding)
        self.act = activation_function
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            input features: (B, L, I_EMBED)
        Return:
            output features: (B, H_EMBED)
        """
        # Check size of tensors
        x = x.transpose(1, 2) # (B, I_EMBED, L)
        x = self.conv(x) # (B, H_EMBED, L)
        x = self.act(x) # (B, H_EMBED, L)
        x = self.dropout(x) # (B, H_EMBED, L)
        x = x.transpose(1, 2) # (B, L, H_EMBED)
        return x

def is_whitespace(char):
    """    Checks whether `chars` is a whitespace character.
        \t, \n, and \r are technically contorl characters but we treat them
        as whitespace since they are generally considered as such.
    """
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control(char):
    """    Checks whether `chars` is a control character.
        These are technically control characters but we count them as whitespace characters.
    """
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def is_punctuation(char):
    """ Checks whether `chars` is a punctuation character.
        We treat all non-letter/number ASCII as punctuation. Characters such as "^", "$", and "`" are not in the Unicode.
        Punctuation class but we treat them as punctuation anyways, for consistency.
    """
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def is_chinese_char(cp):
    """    Checks whether CP is the codepoint of a CJK character.
        This defines a "chinese character" as anything in the CJK Unicode block:
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        despite its name. The modern Korean Hangul alphabet is a different block,
        as is Japanese Hiragana and Katakana. Those alphabets are used to write
        space-separated words, so they are not treated specially and handled
        like the all of the other languages.
    """
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or
        (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True
    return False

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def split_on_whitespace(text):
    """ Runs basic whitespace cleaning and splitting on a peice of text.
    e.g, 'a b c' -> ['a', 'b', 'c']
    """
    text = text.strip()
    if not text:
        return []
    return text.split()

def split_on_punctuation(text):
    """Splits punctuation on a piece of text."""
    start_new_word = True
    output = []
    for char in text:
        if is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
    return ["".join(x) for x in output]

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if is_chinese_char(cp):
            output.append(" ")
            output.append(char)
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)

def strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    if vocab_file ==  None:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    if isinstance(vocab_file, str) or isinstance(vocab_file, bytes):
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab
    else:
        return vocab_file

def printable_text(text):
    """    Returns text encoded in a way suitable for print or `tf.logging`.
        These functions want `str` for both Python2 and Python3, but in one case
        it's a Unicode string and in the other it's a byte string.
    """
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def convert_by_vocab(vocab, items, max_seq_length = None, blank_id = 0, unk_id = 1, uncased = True):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if uncased:
            item = item.lower()
        if item in vocab:
            output.append(vocab[item])
        else:
            output.append(unk_id)
    if max_seq_length != None:
        if len(output) > max_seq_length:
            output = output[:max_seq_length]
        else:
            while len(output) < max_seq_length:
                output.append(blank_id)
    return output

def convert_tokens_to_ids(vocab, tokens, max_seq_length = None, blank_id = 0, unk_id = 1):
    return convert_by_vocab(vocab, tokens, max_seq_length, blank_id, unk_id)

def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

def add_token(tokens_a, tokens_b = None):
    assert len(tokens_a) >= 1
    
    tokens = []
    segment_ids = []
    
    tokens.append("[CLS]")
    segment_ids.append(0)
    
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b != None:
        assert len(tokens_b) >= 1

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)

        tokens.append("[SEP]")
        segment_ids.append(1)

    return tokens, segment_ids

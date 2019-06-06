import re
import numpy as np
from keras.utils import to_categorical
from constants import TIMESTEPS, TEXT_CORPUS_PATH
from utils import read_file


def get_sequences(text, char_to_int, seq_length):
    '''For every char 'c' in text, create input array
    containing 'seq_length' previous chars encoded to integers
    and add 'c' to output array encoded as integer

    Args:
        text (string): Text corpus
        char_to_int (dict): Character to distinct integer mapping
        seq_length (int): Length of chars preceding the target char
    '''
    inputs  = []
    outputs = []

    for i in range(0, len(text) - seq_length, 1):
        sequence_in  = text[i:i + seq_length]
        sequence_out = text[i + seq_length]
        inputs.append([char_to_int[char] for char in sequence_in])
        outputs.append(char_to_int[sequence_out])

    return np.array(inputs), np.array(outputs)


def convert_dataset(raw_text):
    '''Convert raw text corpus to X,y pair. y contains target char
    encoded to integers, X contains 'TIMESTEPS' preceding chars encoded
    to integers corresponding to target char

    Args:
        raw_text (string): Text corpus
    '''
    chars = sorted(list(set(raw_text)))
    char_to_int = {char: index for index, char in enumerate(chars)}
    n_vocab = len(chars)

    seq_inputs, outputs = get_sequences(raw_text, char_to_int, TIMESTEPS)
    n_samples = len(seq_inputs)

    return seq_inputs, to_categorical(outputs), n_vocab, chars


def get_javascript_dataset():
    '''Get JS codes concatenated string file'''
    raw_text = read_file(TEXT_CORPUS_PATH)
    raw_text = re.sub('    ', '\t', raw_text).strip()  # convert 4 spaces to tab
    return convert_dataset(raw_text)

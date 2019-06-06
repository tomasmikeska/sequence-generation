import re
import numpy as np
from keras.utils import to_categorical
from constants import TIMESTEPS, TEXT_CORPUS_PATH


def read_file(filename):
    return open(filename, encoding='utf8').read()


def get_sequences(text, n_chars, char_to_int, seq_length):
    inputs  = []
    outputs = []

    for i in range(0, n_chars - seq_length, 1):
        sequence_in  = text[i:i + seq_length]
        sequence_out = text[i + seq_length]
        inputs.append([char_to_int[char] for char in sequence_in])
        outputs.append(char_to_int[sequence_out])

    return np.array(inputs), np.array(outputs)


def convert_dataset(raw_text):
    chars = sorted(list(set(raw_text)))
    char_to_int = {char: index for index, char in enumerate(chars)}

    n_chars = len(raw_text)
    n_vocab = len(chars)

    seq_inputs, outputs = get_sequences(raw_text, n_chars, char_to_int, TIMESTEPS)
    n_samples = len(seq_inputs)

    return seq_inputs, to_categorical(outputs), n_vocab, chars


def get_javascript_dataset():
    raw_text = read_file(TEXT_CORPUS_PATH)
    raw_text = re.sub('    ', '\t', raw_text).strip()  # convert 4 spaces to tab
    return convert_dataset(raw_text)

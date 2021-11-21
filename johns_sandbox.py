from token import STAR
import numpy as np
from utils.utils import START_TOKEN, STOP_TOKEN, tokenizer

def pad_corpus(ciphered_text, window_size):
    window_size -= 2
    num_windows = ciphered_text.size // window_size
    ciphered_text = ciphered_text[: num_windows * window_size]
    ciphered_text = np.reshape(ciphered_text, (-1, window_size))

    start_tokens = np.ones(num_windows, dtype=np.uint8) * tokenizer[START_TOKEN]
    stop_tokens = np.ones(num_windows, dtype=np.uint8) * tokenizer[STOP_TOKEN]
    ciphered_text = np.transpose(np.concatenate(start_tokens,
                                                np.transpose(ciphered_text),
                                                stop_tokens))

    return ciphered_text

def get_batches(inputs, labels, batch_size):
    input_batches = []
    label_batches = []

    i = 0
    while i < len(inputs):
        input_batch = inputs[i: i + batch_size]
        label_batch = labels[i: i + batch_size]

        input_batches.append(input_batch)
        label_batches.append(label_batch)

        i += batch_size

    return input_batches, label_batches

def load_caesar_data(file_name, window_size):
    data = np.load(file_name)
    ciphers = data["cipher"]
    plain = data["plain"]

    padded_ciphers = pad_corpus(ciphers, window_size)
    padded_plain = pad_corpus(plain, window_size)

    return get_batches(padded_ciphers, padded_plain)

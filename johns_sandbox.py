import numpy as np
import tensorflow as tf
from token import STAR
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
    i = 0
    while i < len(inputs):
        yield inputs[i: i + batch_size], labels[i: i + batch_size]
        i += batch_size

def load_caesar_data(file_name, window_size, batch_size, shuffle=True):
    data = np.load(file_name)
    ciphers = data["cipher"]
    plain = data["plain"]

    padded_ciphers = pad_corpus(ciphers, window_size)
    padded_plain = pad_corpus(plain, window_size)

    if shuffle:
        indices  = tf.range(0, tf.shape(padded_ciphers)[0])
        indices = tf.random.shuffle(indices)

        padded_ciphers = tf.gather(padded_ciphers, indices)
        padded_plain = tf.gather(padded_plain, indices)

    return get_batches(padded_ciphers, padded_plain, batch_size)

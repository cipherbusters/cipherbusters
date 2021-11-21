import numpy as np
import tensorflow as tf
from utils.utils import pad_corpus, get_batches

def load_data(file_name, window_size, batch_size, shuffle=True):
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

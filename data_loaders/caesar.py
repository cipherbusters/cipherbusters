import numpy as np
import tensorflow as tf
from utils.utils import pad_corpus, get_batches


def load_data(file_name, window_size, batch_size, shuffle=True, use_pct=1.0, train_pct=0.8):
    data = np.load(file_name)

    # load data and cut out unused data
    data_len = len(data['cipher'])
    ciphers = data["cipher"][:int(data_len * use_pct)]
    plain = data["plain"][:int(data_len * use_pct)]

    padded_ciphers = tf.convert_to_tensor(pad_corpus(ciphers, window_size))
    padded_plain = tf.convert_to_tensor(pad_corpus(plain, window_size))

    if shuffle:
        indices = tf.range(0, tf.shape(padded_ciphers)[0])
        indices = tf.random.shuffle(indices)

        padded_ciphers = tf.gather(padded_ciphers, indices)
        padded_plain = tf.gather(padded_plain, indices)

    train_ciphers = padded_ciphers[:int(len(padded_ciphers) * train_pct)]
    test_ciphers = padded_ciphers[int(len(padded_ciphers) * train_pct):]
    train_plain = padded_plain[:int(len(padded_ciphers) * train_pct)]
    test_plain = padded_plain[int(len(padded_ciphers) * train_pct):]

    return get_batches(train_ciphers, train_plain, batch_size), get_batches(test_ciphers, test_plain, batch_size)

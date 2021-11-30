import numpy as np
import tensorflow as tf
import random
from utils.utils import pad_corpus, get_substitution_batches, tokenize

def load_data(plaintext_file, window_size, batch_size, batch_limit=3000, add_startstop=False, shuffle=True, use_pct=1.0, train_pct=0.8):
    plaintext = []
    with open(plaintext_file, 'r') as f:
        for line in f:
            pt = line[:-1] + ' '
            plaintext.append(list(pt))
    plaintext = np.concatenate(plaintext, 0)

    # load data and cut out unused data
    plaintext = plaintext[:int(len(plaintext) * use_pct)]
    windowed_plain = pad_corpus(plaintext, window_size, add_startstop)

    if shuffle:
        random.shuffle(windowed_plain)

    train_plain = windowed_plain[:int(len(windowed_plain) * train_pct)]
    test_plain = windowed_plain[int(len(windowed_plain) * train_pct):]

    return get_substitution_batches(train_plain, batch_size, batch_limit), get_substitution_batches(test_plain, batch_size, batch_limit)
import numpy as np
import tensorflow as tf
import random
from dataloaders.dataloader import Dataloader
from utils.utils import pad_corpus, get_vigenere_batches


class VigenereDataset(Dataloader):
    def __init__(self, batch_size, window_size, train_pct=0.8):
        self.batch_size = batch_size
        self.window_size = window_size
        self.train_pct = train_pct
        self.load_data()

    def load_data(self):
        plaintext = []
        with open(self.DATA_DIR / 'clean.txt', 'r') as f:
            for line in f:
                pt = line[:-1] + ' '
                plaintext.append(list(pt))
        plaintext = np.concatenate(plaintext, 0)
        windowed_plain = pad_corpus(plaintext, self.window_size, False)
        random.shuffle(windowed_plain)
        self.train_plain = windowed_plain[:int(
            len(windowed_plain) * self.train_pct)]
        self.test_plain = windowed_plain[int(
            len(windowed_plain) * self.train_pct):]

    def get_train_epoch(self):
        return get_vigenere_batches(self.train_plain, self.batch_size)

    def get_train_len(self):
        return len(self.train_plain) // self.batch_size

    def get_test_epoch(self):
        return get_vigenere_batches(self.test_plain, self.batch_size)

    def get_test_len(self):
        return len(self.test_plain) // self.batch_size

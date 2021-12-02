import numpy as np
import tensorflow as tf
from utils.utils import pad_corpus, get_batches
from dataloaders.dataloader import Dataloader


class CaesarDataset(Dataloader):
    def __init__(self, files, batch_size, window_size, train_pct=0.8):
        self.batch_size = batch_size
        self.window_size = window_size
        self.train_pct = train_pct
        self.files = files
        self.load_data()

    def load_data(self):
        train_dataloader = []
        test_dataloader = []
        for file in self.files:
            train, test = self.load_file(
                self.DATA_DIR / f'caesar_{file}.npz', use_pct=1 / len(self.files))
            train_dataloader += list(train)
            test_dataloader += list(test)
        # shuffle
        train_len = len(train_dataloader)
        test_len = len(test_dataloader)
        train_indices = tf.random.shuffle(tf.range(0, train_len))
        train_dataloader = tf.gather(train_dataloader, train_indices)
        test_indices = tf.random.shuffle(tf.range(0, test_len))
        test_dataloader = tf.gather(test_dataloader, test_indices)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_train_epoch(self):
        for batch in self.train_dataloader:
            yield batch

    def get_train_len(self):
        return len(self.train_dataloader)

    def get_test_epoch(self):
        for batch in self.test_dataloader:
            yield batch

    def get_test_len(self):
        return len(self.test_dataloader)

    def load_file(self, file_name, use_pct=1.0):
        data = np.load(file_name)

        # load data and cut out unused data
        data_len = len(data['cipher'])
        ciphers = data["cipher"][:int(data_len * use_pct)]
        plain = data["plain"][:int(data_len * use_pct)]

        padded_ciphers = tf.convert_to_tensor(
            pad_corpus(ciphers, self.window_size, False))
        padded_plain = tf.convert_to_tensor(
            pad_corpus(plain, self.window_size, False))

        train_ciphers = padded_ciphers[:int(
            len(padded_ciphers) * self.train_pct)]
        test_ciphers = padded_ciphers[int(
            len(padded_ciphers) * self.train_pct):]
        train_plain = padded_plain[:int(len(padded_ciphers) * self.train_pct)]
        test_plain = padded_plain[int(len(padded_ciphers) * self.train_pct):]

        return get_batches(train_ciphers, train_plain, self.batch_size), get_batches(test_ciphers, test_plain, self.batch_size)

import numpy as np
import tensorflow as tf


class RNN(tf.keras.Model):
    def __init__(self, alphabet_size, embedding_size, window_size):
        super(RNN, self).__init__()
        self.window_size = window_size
        self.alphabet_size = alphabet_size

        self.emb_ciphertext = tf.keras.layers.Embedding(
            self.alphabet_size, self.embedding_size)
        self.emb_plaintext = tf.keras.layers.Embedding(
            self.alphabet_size, self.embedding_size)

        self.encoder = tf.keras.layers.LSTM(
            self.embedding_size, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.LSTM(
            self.embedding_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(
            self.alphabet_size, activation='softmax')

    def call(self, ciphertext, plaintext):
        # embed the ciphertext characters
        ciphertext_embs = self.emb_ciphertext(ciphertext)

        # encode the ciphertext in a cell_state/hidden_state
        _, cell_state, hidden_state = self.encoder(
            ciphertext_embs, initial_state=None)

        # embed the plaintext characters
        plaintext_embs = self.emb_plaintext(plaintext)

        # decode the plaintext given an encoding of the ciphertext by teacher forcing
        decoded, _, _ = self.decoder(
            plaintext_embs, initial_state=(cell_state, hidden_state))
        probs = self.dense(decoded)

        return probs

    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

import numpy as np
import tensorflow as tf
from utils.utils import START_TOKEN, detokenizer, STOP_TOKEN, tokenize, tokenizer

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
    
    def decode(self, cipher):
        cipher = tf.convert_to_tensor(np.concatenate([[tokenizer[START_TOKEN]], tokenize(cipher)]))
        cipher = tf.expand_dims(cipher, axis=0)
        cipher = self.emb_ciphertext(cipher)
        # encode the ciphertext in a cell_state/hidden_state
        _, cell_state, hidden_state = self.encoder(
            cipher, initial_state=None)
        output = []
        decoded_char = START_TOKEN
        while decoded_char != STOP_TOKEN:
            decoded = tf.expand_dims([tokenizer[decoded_char]], axis=0)
            decoded = self.emb_plaintext(decoded)
            decoded, cell_state, hidden_state = self.decoder(decoded, initial_state=(cell_state, hidden_state))
            probs = self.dense(decoded)
            decoded_char = detokenizer[tf.argmax(probs, axis=2).numpy().item()]
            output.append(decoded_char)
        return ''.join(output[:-1])

    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

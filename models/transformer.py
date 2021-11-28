import numpy as np
import tensorflow as tf
import models.transformer_utils as utils

class Transformer(tf.keras.Model): 
    def __init__(self, window_size, alphabet_size, embedding_size):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size

        self.emb_ciphertext = tf.keras.layers.Embedding(self.alphabet_size, self.embedding_size)
        self.emb_plaintext = tf.keras.layers.Embedding(self.alphabet_size, self.embedding_size)

        # TODO: left out positional encoding layer. re-evaluate later.

        self.encoder = utils.Transformer_Block(self.embedding_size, is_decoder=False)
        self.decoder = utils.Transformer_Block(self.embedding_size, is_decoder=True)
        self.fully_connected = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.alphabet_size, activation='softmax')
        ])
    
    def call(self, ciphertext, plaintext):
        # embed the ciphertext characters
        ciphertext_embs = self.emb_ciphertext(ciphertext)

        # encode the ciphertext using a transformer
        enc_out = self.encoder(ciphertext_embs)

        # embed the plaintext characters
        plaintext_embs = self.emb_plaintext(plaintext)

        # decode the plaintext given an encoding of the ciphertext
        dec_out = self.decoder(plaintext_embs, context=enc_out)

        probs = self.fully_connected(dec_out)
        return probs
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy
    
    def loss(self, probs, labels):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
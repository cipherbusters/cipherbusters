import numpy as np
import tensorflow as tf
import models.transformer_utils as utils
from models.base import Base
import warnings
warnings.filterwarnings("ignore")

class Beefy_Transformer(Base): 
    def __init__(self, alphabet_size):
        super(Beefy_Transformer, self).__init__(alphabet_size)
        # TODO: left out positional encoding layer. re-evaluate later.
        self.encoder = tf.keras.Sequential([
            utils.Transformer_Block(self.alphabet_size, is_decoder=False),
            utils.Transformer_Block(self.alphabet_size, is_decoder=False)
        ])
        self.decoder1 = utils.Transformer_Block(self.alphabet_size, is_decoder=True)
        self.decoder2 = utils.Transformer_Block(self.alphabet_size, is_decoder=False)
        self.fully_connected = tf.keras.Sequential([
            tf.keras.layers.Dense(2**11, activation='relu'),
            tf.keras.layers.Dense(2**8, activation='relu'),
            tf.keras.layers.Dense(self.alphabet_size, activation='softmax')
        ])
    
    def call(self, ciphertext):
        # embed the ciphertext characters
        ciphertext = tf.one_hot(ciphertext, self.alphabet_size)

        # encode the ciphertext using a transformer
        enc_out = self.encoder(ciphertext)

        # decode the plaintext given an encoding of the ciphertext
        dec_out = self.decoder2(self.decoder1(ciphertext, context=enc_out))

        probs = self.fully_connected(dec_out)
        return probs
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy
    
    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
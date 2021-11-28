import numpy as np
import tensorflow as tf
from models.base import Base

class RNN(Base):
    def __init__(self, alphabet_size):
        super().__init__(alphabet_size)

        self.encoder = tf.keras.layers.LSTM(self.alphabet_size, return_state=True)
        self.decoder = tf.keras.layers.LSTM(self.alphabet_size, return_sequences=True)
        self.fully_connected = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.alphabet_size, activation='softmax')
        ])

    def call(self, ciphertext):
        # embed the ciphertext using one-hot encodings
        ciphertext = tf.one_hot(ciphertext, self.alphabet_size)
        
        # pass the ciphertext through an lstm, producing output at each time step
        _, state_h, state_c = self.encoder(ciphertext)
        out = self.decoder(ciphertext, initial_state=(state_h, state_c))
        
        # run sequential output through dense layer to get probability distribution over alphabet
        probs = self.fully_connected(out)
        
        return probs
        

    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy

    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))


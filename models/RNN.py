import numpy as np
import tensorflow as tf

class RNN(tf.keras.Model): 
    def __init__(self, window_size, alphabel_size):
        super(RNN, self).__init__()
        self.window_size = window_size
        self.alphabel_size = alphabel_size
        self.learning_rate = 1e-3
        self.batch_size = 128
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.encoder = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dense(self.alphabet_size, activation='softmax')
        ])
    
    def call(self, ciphertext, plaintext):
        # encode the ciphertext in a cell_state/hidden_state
        _, cell_state, hidden_state = self.encoder(ciphertext, initial_state=None)
        
        # decode the plaintext given an encoding of the ciphertext by teacher forcing
        probs = self.decoder(plaintext, initial_state=(cell_state, hidden_state))
        
        return probs
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))
        return accuracy
    
    def loss(self, probs, labels):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))


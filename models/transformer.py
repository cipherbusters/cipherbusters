import numpy as np
import tensorflow as tf

class Transformer(tf.keras.Model): 
    def __init__(self, window_size, alphabel_size):
        super(RNN, self).__init__()
        self.window_size = window_size
        self.alphabel_size = alphabel_size
        self.learning_rate = 1e-3
        self.batch_size = 128
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        pass
    
    def call(self, ciphertext, plaintext):
        pass
    
    def accuracy(self, probs, labels):
        pass
    
    def loss(self, probs, labels):
        pass
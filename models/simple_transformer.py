import numpy as np
import tensorflow as tf
import models.transformer_utils as utils
from models.base import Base
class Simple_Transformer(Base): 
    def __init__(self, alphabet_size):
        super().__init__(alphabet_size)

        # TODO: left out positional encoding layer. re-evaluate later.

        self.transformer = utils.Transformer_Block(self.alphabet_size, is_decoder=False)
        self.dense = tf.keras.layers.Dense(self.alphabet_size, activation='softmax')
    
    def call(self, ciphertext):
        # embed the ciphertext using one-hot encodings
        ciphertext = tf.one_hot(ciphertext, self.alphabet_size)
        
        # pass the ciphertext through an lstm, producing output at each time step
        out = self.transformer(ciphertext)
        
        # run sequential output through dense layer to get probability distribution over alphabet
        probs = self.dense(out)
        
        return probs
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy
    
    def loss(self, probs, labels):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
import numpy as np
import tensorflow as tf
import models.transformer_utils as utils
from models.base import Base

class Discriminator(tf.keras.layers.Layer):
    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.encoder = utils.Transformer_Block(self.alphabet_size, is_decoder=False)
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

    @tf.function
    def call(self, x):
        x = tf.one_hot(x, self.alphabet_size)
        out = self.encoder(x)
        out = self.dense1(tf.reduce_mean(out, axis=1)) # average along the window dimension
        return self.dense2(out)
    
class BeefyTransformer(tf.keras.layers.Layer):
    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        
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
    
    @tf.function
    def call(self, ciphertext):
        # embed the ciphertext characters
        ciphertext = tf.one_hot(ciphertext, self.alphabet_size)

        # encode the ciphertext using a transformer
        enc_out = self.encoder(ciphertext)

        # decode the plaintext given an encoding of the ciphertext
        dec_out = self.decoder2(self.decoder1(ciphertext, context=enc_out))

        probs = self.fully_connected(dec_out)
        return probs

class BeefyAdversarialTransformer(Base):
    def __init__(self, alphabet_size):
        super().__init__(alphabet_size)
        self.transformer = BeefyTransformer(alphabet_size)
        self.discriminator = Discriminator(alphabet_size)
        
    def call(self, ciphertext):
        return self.transformer(ciphertext)
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = np.mean(pred == tf.cast(labels, dtype=tf.int64))
        return accuracy
    
    def loss_vanilla(self, probs, labels):
        self.discriminator.trainable = False
        self.transformer.trainable = True
        
        decryption_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
        return decryption_loss
    
    def loss_G(self, probs):
        pred = tf.argmax(probs, axis=-1)
        probs_false = tf.stop_gradient(self.discriminator(pred))
        shape_labels = (len(probs), )
        discriminator_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.zeros(shape_labels), probs_false))
        return discriminator_loss
    
    def loss_D(self, pred, labels):
        self.discriminator.trainable = True
        self.transformer.trainable = False
        
        probs_true = self.discriminator(tf.stop_gradient(labels))
        probs_false = self.discriminator(tf.stop_gradient(pred))
        
        shape_labels = (len(probs_true), )
        
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.zeros(shape_labels), probs_true)) + \
            tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.ones(shape_labels), probs_false))
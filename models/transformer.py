import numpy as np
import tensorflow as tf
import transformer_utils

class Transformer(tf.keras.Model): 
    def __init__(self, window_size, alphabel_size, embedding_size):
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.alphabel_size = alphabel_size
        self.embedding_size = embedding_size
        self.learning_rate = 1e-3
        self.batch_size = 128
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.emb_ciphertext = tf.keras.layers.Embedding(self.alphabet_size, self.embedding_size)
        self.emb_plaintext = tf.keras.layers.Embedding(self.alphabet_size, self.embedding_size)

        # TODO: left out positional encoding layer. re-evaluate later.

        self.encoder = transformer_utils.Transformer_Block(self.embedding_size, is_decoder=False)
        self.decoder = tf.keras.Sequential([
            transformer_utils.Transformer_Block(self.embedding_size, is_decoder=True),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.alphabel_size, activation='softmax')
        ])
    
    def call(self, ciphertext, plaintext):
        # embed the ciphertext characters
        ciphertext_embs = self.fr_embs(ciphertext)

        # encode the ciphertext using a transformer
        enc_out = self.encoder(ciphertext_embs)

        # embed the plaintext characters
        plaintext_embs = self.eng_embs(plaintext)

        # decode the plaintext given an encoding of the ciphertext
        probs = self.decoder(plaintext_embs, context=enc_out)

        return probs
    
    def accuracy(self, probs, labels):
        pred = tf.argmax(input=probs, axis=2)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, labels), dtype=tf.float32))
        return accuracy
    
    def loss(self, probs, labels):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))
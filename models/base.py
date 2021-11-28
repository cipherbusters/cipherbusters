import tensorflow as tf
from utils.utils import detokenize, tokenize
class Base(tf.keras.Model):
    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
    
    def decrypt(self, text):
        text = tokenize(text)
        text = tf.expand_dims(text, axis=0)
        probs = self.call(text)[0]
        return detokenize(tf.argmax(probs, axis=-1))
from pathlib import Path
import tensorflow as tf
import numpy as np
from .ciphers import generate_substitution, substitution_encode, generate_key, vigenere_encode
DATA_DIR = Path(__file__).parent.parent / 'data'


def build_tokens():
    tokenizer = {}
    detokenizer = {}
    for i, c in enumerate(SPECIAL_TOKENS + list('abcdefghijklmnopqrstuvwxyz0123456789 ')):
        tokenizer[c] = i
        detokenizer[i] = c
    return tokenizer, detokenizer


START_TOKEN = '<start>'
STOP_TOKEN = '<stop>'
SPECIAL_TOKENS = [START_TOKEN, STOP_TOKEN]
tokenizer, detokenizer = build_tokens()


def tokenize(s: str):
    output = []
    for c in s:
        output.append(tokenizer[c])
    return np.array(output, dtype=np.uint8)


def detokenize(t) -> str:
    characters = tf.reshape(tf.convert_to_tensor(t), -1).numpy()
    decoded = []
    for c in characters:
        decoded_c = detokenizer[c]
        if decoded_c not in SPECIAL_TOKENS:
            decoded.append(decoded_c)
    return ''.join(decoded)


def pad_corpus(ciphered_text, window_size, add_startstop=False):
    if add_startstop:
        window_size -= 2
    num_windows = ciphered_text.size // window_size
    ciphered_text = ciphered_text[: num_windows * window_size]
    ciphered_text = np.reshape(ciphered_text, (-1, window_size))
    if not add_startstop:
        return ciphered_text
    start_tokens = np.ones(
        (1, num_windows), dtype=np.uint8) * tokenizer[START_TOKEN]
    stop_tokens = np.ones(
        (1, num_windows), dtype=np.uint8) * tokenizer[STOP_TOKEN]
    ciphered_text = np.transpose(np.concatenate([start_tokens,
                                                np.transpose(ciphered_text),
                                                stop_tokens], axis=0))
    return ciphered_text


def get_batches(inputs, labels, batch_size):
    i = 0
    while i + batch_size < len(inputs):
        yield inputs[i: i + batch_size], labels[i: i + batch_size]
        i += batch_size
    
def get_substitution_batches(plaintext, batch_size, batch_limit):
    for _ in range(batch_limit):
        sub = generate_substitution()
        i = np.random.randint(len(plaintext)-batch_size, size=1)[0]
        inputs = []
        labels = []
        for s in plaintext[i:i+batch_size]:
            inputs.append(tokenize(substitution_encode(s, sub)))
            labels.append(tokenize(s))
        inputs = np.array(inputs)
        labels = np.array(labels)
        yield inputs, labels

def get_vigenere_batches(plaintext, batch_size, batch_limit):
    for _ in range(batch_limit):
        key = generate_key()
        i = np.random.randint(len(plaintext)-batch_size, size=1)[0]
        inputs = []
        labels = []
        for s in plaintext[i:i+batch_size]:
            inputs.append(tokenize(vigenere_encode(s, key)))
            labels.append(tokenize(s))
        inputs = np.array(inputs)
        labels = np.array(labels)
        yield inputs, labels

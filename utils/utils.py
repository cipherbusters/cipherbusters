from pathlib import Path
import tensorflow as tf
import numpy as np
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
    characters = tf.reshape(tf.convert_to_tensor(t), -1).tolist()
    decoded = []
    for c in characters:
        decoded_c = detokenizer[c]
        if decoded_c not in SPECIAL_TOKENS:
            decoded.append(decoded_c)
    return ''.join(decoded)

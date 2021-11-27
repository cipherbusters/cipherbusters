from os import EX_CANTCREAT
import random

def build_vocab():
    vocab = {}
    decoder = {}
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789'):
        vocab[c] = i
        decoder[i] = c
    return vocab, decoder

VOCAB, DECODER = build_vocab()

def caesar_encode(s: str, k: int) -> str:
    outputs = []
    for c in s:
        if c in VOCAB:
            outputs.append(
                DECODER[(VOCAB[c] + k) % len(VOCAB)])
        else:
            outputs.append(c)
    return ''.join(outputs)


# SUBSTITUTION
def generate_substitutions():
    subs = []
    for _ in range(100):
        keys = list(VOCAB.keys())
        vals = keys.copy()
        random.shuffle(vals)

        sub = dict(zip(keys, vals))
        subs.append(sub)

    return subs

SUBSTITUTIONS = generate_substitutions()

def substitution_encode(s, substitution):
    encryption = []
    for plain_c in s:
        if VOCAB.has_key(plain_c):
            sub_c = substitution[plain_c]
            encryption.append(sub_c)
        else:
            encryption.append(plain_c)
        
    return "".join(encryption)


# VIGENERE

# def build_matrix():
#     matrix = []
#     for c in VOCAB:


# def vigenere_encode(s, key):
#     encryption = []
#     for i, plain_c in enumerate(s):
#         if plain_c in VOCAB:
#             vig_c = (ord(plain_c) + key[i % len(key)]) % VOCAB_SIZE
#             vig_c += ord("A")


#         else:
#             encryption.append(plain_c)

#     return "".join(encryption)

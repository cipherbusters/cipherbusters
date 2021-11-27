from utils.utils import DATA_DIR, tokenize
from utils.ciphers import caesar_encode, substitution_encode, VOCAB, SUBSTITUTIONS
import tensorflow_datasets as tfds
import argparse
from tqdm import tqdm
import tensorflow as tf
import numpy as np
dataset = tfds.load('wikipedia/20201201.en', split='train')


VOCAB = set(VOCAB.keys())
SPACES = set([' ', '\n'])


def clean(s: str) -> str:
    output = []
    for c in s.lower():
        if c in SPACES:
            if len(output) > 0 and output[-1] != ' ':
                output.append(' ')
        elif c in VOCAB:
            output.append(c)
    if len(output) > 0 and output[0] == ' ':
        output = output[1:]
    if len(output) > 0 and output[-1] == ' ':
        output = output[:-1]
    return ''.join(output)


def generate(args):
    with open(DATA_DIR / 'clean.txt', 'w') as f:
        for entry in tqdm(dataset.take(args.n), total=args.n):
            f.write(clean(entry['text'].numpy().decode('utf-8')))
            f.write('\n')


def generate_caesar(args):
    progbar = tqdm(total=args.n * len(VOCAB))
    for k in range(len(VOCAB)):
        tokenized_plain = []
        tokenized_cipher = []
        with open(DATA_DIR / 'clean.txt', 'r') as f:
            for line in f:
                plaintext = line[:-1] + ' '
                ciphertext = caesar_encode(plaintext, k)
                tokenized_plain.append(tokenize(plaintext))
                tokenized_cipher.append(tokenize(ciphertext))
                progbar.update()
        tokenized_plain = np.concatenate(tokenized_plain, 0)
        tokenized_cipher = np.concatenate(tokenized_cipher, 0)
        np.savez(DATA_DIR / f'caesar_{k}',
                 plain=tokenized_plain, cipher=tokenized_cipher)
    progbar.clear()


def generate_substitution(args):
    progbar = tqdm(total=args.n * len(VOCAB))

    for i, sub in enumerate(SUBSTITUTIONS):
        tokenized_plain = []
        tokenized_cipher = []
        with open(DATA_DIR / 'clean.txt', 'r') as f:
            for line in f:
                plaintext = line[:-1] + ' '
                ciphertext = substitution_encode(plaintext, sub)
                tokenized_plain.append(tokenize(plaintext))
                tokenized_cipher.append(tokenize(ciphertext))
                progbar.update()
        tokenized_plain = np.concatenate(tokenized_plain, 0)
        tokenized_cipher = np.concatenate(tokenized_cipher, 0)
        np.savez(DATA_DIR / f'substitution_{i}',
                 plain=tokenized_plain, cipher=tokenized_cipher)
    progbar.clear()


def main(args):
    if args.action == 'generate':
        generate(args)
    elif args.action == 'caesar':
        generate_caesar(args)
    elif args.action == 'substitution':
        generate_substitution(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'caesar', 'substitution'])
    parser.add_argument('-n', type=int, default=20000)

    args = parser.parse_args()

    main(args)
    
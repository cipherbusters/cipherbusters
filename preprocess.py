from utils.utils import DATA_DIR
from utils.ciphers import caesar_encode, CAESAR_VOCAB
import tensorflow_datasets as tfds
import argparse
from tqdm import tqdm
dataset = tfds.load('wikipedia/20201201.en', split='train')


VOCAB = set(CAESAR_VOCAB.keys())
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
    with open(DATA_DIR / 'caesar.txt', 'w') as fout:
        with open(DATA_DIR / 'clean.txt', 'r') as fin:
            for line in tqdm(fin, total=args.n):
                plaintext = line[:-1]
                ciphertext = caesar_encode(plaintext, args.k)
                fout.write(ciphertext + '\n')
                fout.write(plaintext + '\n')


def main(args):
    if args.action == 'generate':
        generate(args)
    elif args.action == 'caesar':
        generate_caesar(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['generate', 'caesar'])
    parser.add_argument('--n', type=int, default=1000000)
    parser.add_argument('--k', type=int, default=3)

    args = parser.parse_args()

    main(args)

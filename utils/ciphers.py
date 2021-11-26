def build_vocab():
    vocab = {}
    decoder = {}
    for i, c in enumerate('abcdefghijklmnopqrstuvwxyz0123456789'):
        vocab[c] = i
        decoder[i] = c
    return vocab, decoder


CAESAR_VOCAB, CAESAR_DECODER = build_vocab()


def caesar_encode(s: str, k: int) -> str:
    outputs = []
    for c in s:
        if c in CAESAR_VOCAB:
            outputs.append(
                CAESAR_DECODER[(CAESAR_VOCAB[c] + k) % len(CAESAR_VOCAB)])
        else:
            outputs.append(c)
    return ''.join(outputs)
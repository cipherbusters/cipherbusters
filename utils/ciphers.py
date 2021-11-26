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

SUB_1 = dict(zip(list("abcdefghijklmnopqrstuvwxyz0123456789"),
                 list("1234567890qwertyuiopasdfghjklzxcvbnm")))

def substitution_encode(s, substitution):
    outputs = []
    for plain_c in s:
        if plain_c in VOCAB:
            sub_c = substitution[plain_c]
            outputs.append(sub_c)
        else:
            outputs.append(plain_c)
    return "".join(outputs)

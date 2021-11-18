import numpy as np

def bust(ciphered_text, window_size):
    num_windows = ciphered_text.size // window_size
    ciphered_text = ciphered_text[: num_windows * window_size]
    ciphered_text = np.reshape(ciphered_text, (-1, window_size))

    for window in ciphered_text:
        window = ["*START*"] + window + ["*STOP*"]

    return ciphered_text

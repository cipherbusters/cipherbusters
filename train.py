from utils.utils import DATA_DIR, tokenizer, detokenize
from utils.ciphers import CAESAR_VOCAB
import tensorflow as tf
import numpy as np
from models.RNN import RNN
from models.transformer import Transformer
from tqdm import tqdm
import argparse
from data_loaders.caesar import load_data
from pathlib import Path
import matplotlib.pyplot as plt

CKPT_DIR = Path(__file__).parent / 'checkpoints'

# for testing/debugging purposes to train models faster
REDUCTION_FACTOR = 10

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_transformer", action="store_true")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=int, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=5)
    args = parser.parse_args()
    return args


def train(model, dataloader, optimizer):
    pbar = tqdm(dataloader, total=len(dataloader))
    loss_list = []
    for i, (ciphertext, plaintext) in enumerate(pbar):
        #if i % REDUCTION_FACTOR != 0: continue  # TODO: comment out during actual evaluation
        with tf.GradientTape() as tape:
            probs = model(ciphertext[:, 1:], plaintext[:, :-1])
            loss = model.loss(probs, plaintext[:, 1:])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_list.append(loss)
        acc = model.accuracy(probs, plaintext[:, 1:])
        pbar.set_description(f'Loss: {loss.numpy().item():.2f} Accuracy: {acc.item():.2f}')
    return loss_list

def test(model, dataloader):
    print("TESTING ------------------")
    acc_list = []
    for i, (ciphertext, plaintext) in enumerate(dataloader):
        #if i % REDUCTION_FACTOR != 0: continue  # TODO: comment out during actual evaluation
        probs = model(ciphertext[:, 1:], plaintext[:, :-1])
        acc = model.accuracy(probs, plaintext[:, 1:])
        acc_list.append(acc)
    print(f"Accuracy: {np.mean(acc_list)}")


def main(args):
    if args.is_transformer:
        model = Transformer(args.window_size, len(
            tokenizer), args.embedding_size)
    else:
        model = RNN(len(tokenizer), args.embedding_size,
                    args.window_size)

    # load in ciphers
    caesar_ciphers = np.arange(len(CAESAR_VOCAB))
    np.random.shuffle(caesar_ciphers)

    for i in range(len(caesar_ciphers)):
        ciphers = caesar_ciphers[:(i+1)]
        print(f'Training ciphers: {ciphers}')
        train_dataloader = []
        test_dataloader = []
        for c in ciphers:
            tr, te = load_data(
                DATA_DIR / f'caesar_{c}.npz', args.window_size, args.batch_size, use_pct=1 / 5 / len(ciphers))
            train_dataloader += list(tr)
            test_dataloader += list(te)

        train_len = len(train_dataloader)
        test_len = len(test_dataloader)
        train_indices = tf.random.shuffle(tf.range(0, train_len))
        train_dataloader = tf.gather(train_dataloader, train_indices)
        test_indices = tf.random.shuffle(tf.range(0, test_len))
        test_dataloader = tf.gather(test_dataloader, test_indices)

        loss_list = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        for e in range(args.num_epochs):
            print(f'EPOCH {e} of {args.num_epochs} ------------------')
            loss_list += train(model, train_dataloader, optimizer)

            checkpoint_path = CKPT_DIR / 'caesar_rnn' / \
                f'{"+".join(map(lambda x: str(x), ciphers))}' / \
                f'{e:04d}.ckpt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(checkpoint_path)

        # TODO: uncomment to display loss plots 
        #plt.plot(loss_list)
        #plt.show()
        
        test(model, test_dataloader)


if __name__ == '__main__':
    args = parseArguments()
    main(args)

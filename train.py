from utils.utils import get_batches, DATA_DIR, tokenizer
import tensorflow as tf
import numpy as np
from models.RNN import RNN
from models.transformer import Transformer
import sys
import argparse
from data_loaders.caesar import load_data

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_rnn", action="store_true")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--embedding_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    return args

def train(model, dataloader):
    loss = []
    for (ciphertext, plaintext) in dataloader:
        with tf.GradientTape() as tape:
            probs = model(ciphertext, plaintext)
            loss.append(model.loss(probs, plaintext))
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return np.sum(loss)

def test(model, dataloader):
    acc = []
    for (ciphertext, plaintext) in dataloader:
        with tf.GradientTape() as tape:
            probs = model(ciphertext, plaintext)
            acc.append(model.accuracy(probs, plaintext))
        gradients = tape.gradient(acc, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return np.mean(acc)

def main(args):	
    # TODO: Implement dataloader
	# train_dataloader, test_dataloader, alphabet = None, None, None

    if args.is_rnn:
        model = RNN(len(tokenizer), args.embedding_size, args.window_size)
    else:
        model = Transformer() 

	num_epochs = 1
    train_dataloader, test_dataloader = load_data(DATA_DIR / 'caesar_3.npz', 
                                                  args.window_size, 
                                                  args.batch_size)

    for _ in range(num_epochs):
        train(model, train_dataloader)
	# TODO: Set up the testing steps
	acc = test(model, test_dataloader)
    print(f"Accuracy: {acc}")

if __name__ == '__main__':
    args = parseArguments()
    main(args)

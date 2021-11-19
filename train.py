from utils.utils import get_batches
import tensorflow as tf
import numpy as np
from RNN import RNN
from transformer import Transformer
import sys
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_rnn", action="store_true")
    parser.add_argument("--window_size", action="store_true")
    parser.add_argument("--alphabet_size", action="store_true")
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
        model = RNN()
    else:
        model = Transformer() 

	num_epochs = 1
	for _ in range(num_epochs):
        train(model, train_dataloader)

	# TODO: Set up the testing steps
	acc = test(model, test_dataloader)

	pass

if __name__ == '__main__':
    args = parseArguments()
	main(args)
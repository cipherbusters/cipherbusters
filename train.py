from utils.utils import DATA_DIR, tokenizer, detokenize
import tensorflow as tf
import numpy as np
from models.RNN import RNN
from models.transformer import Transformer
from tqdm import tqdm
import argparse
from data_loaders.caesar import load_data
from pathlib import Path

CKPT_DIR = Path(__file__).parent.parent / 'checkpoints'

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--is_transformer", action="store_true")
	parser.add_argument("--window_size", type=int, default=20)
	parser.add_argument("--embedding_size", type=int, default=100)
	parser.add_argument("--learning_rate", type=int, default=1e-3)
	parser.add_argument("--batch_size", type=int, default=128)
	parser.add_argument("--num_epochs", type=int, default=10)
	args = parser.parse_args()
	return args

def train(model, dataloader):
    loss_list = []
    for i, (ciphertext, plaintext) in enumerate(dataloader):
        # reduce amount of training data by factor of 20
        if i % 20 != 0: continue 
        with tf.GradientTape() as tape:
            probs = model(ciphertext[:,1:], plaintext[:,:-1])
            loss = model.loss(probs, plaintext[:,1:])
            acc = model.accuracy(probs, plaintext[:,1:])
            loss_list.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if i % 50 == 0: print(f"batch {i}\tloss: {np.round(loss,2)}\taccuracy: {np.round(acc,2)}")
    return np.mean(loss_list)

def test(model, dataloader):
    print("TESTING ------------------")
    acc_list = []
    for i, (ciphertext, plaintext) in enumerate(dataloader):
        # reduce amount of testing data by factor of 20
        if i % 20 != 0: continue 
        with tf.GradientTape() as tape:
            probs = model(ciphertext[:,1:], plaintext[:,:-1])
            acc = model.accuracy(probs, plaintext[:,1:])
            acc_list.append(acc)
            pred = tf.argmax(input=probs, axis=2)
        if i % 50 == 0: 
            print(f"""batch {i}
                \tacc: {np.round(acc,2)}
                \tplaintext: {detokenize(plaintext[0,1:].numpy())}
                \tprediction: {detokenize(pred[0].numpy())}""")
    return np.mean(acc_list)

def main(args):	
    if args.is_transformer:
        model = Transformer(args.window_size, len(tokenizer), args.embedding_size) 
    else:
        model = RNN(len(tokenizer), args.embedding_size, args.window_size)
    
    caesar_ciphers = [3, 11, 17, 21, 27]
    train_dataloader = []
    test_dataloader = []
    for c in caesar_ciphers:
        tr, te = load_data(DATA_DIR / f'caesar_{c}.npz', args.window_size, args.batch_size)
        train_dataloader += tr
        test_dataloader += te
    
    # shuffle train and test data
    train_len = len(list(train_dataloader))
    test_len = len(list(test_dataloader))
    train_indices  = tf.random.shuffle(tf.range(0, train_len))
    train_dataloader = tf.gather(train_dataloader, train_indices)
    test_indices  = tf.random.shuffle(tf.range(0, test_len))
    test_dataloader = tf.gather(test_dataloader, test_indices)
    
    num_epochs = 1
    for e in range(num_epochs):
        print(f"EPOCH {e} ------------------")
        train(model, train_dataloader)

    acc = test(model, test_dataloader)
    print(f"Accuracy: {acc}")

if __name__ == '__main__':
	args = parseArguments()
	main(args)

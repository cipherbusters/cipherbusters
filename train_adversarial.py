from utils.utils import DATA_DIR, tokenizer, detokenize
from utils.ciphers import VOCAB
import tensorflow as tf
import numpy as np
from models.RNN import RNN
from models.simple_RNN import Simple_RNN
from models.transformer import Transformer
from models.adversarial_transformer import AdversarialTransformer
from tqdm import tqdm
import argparse
from data_loaders.caesar import load_data
from pathlib import Path
import matplotlib.pyplot as plt

CKPT_DIR = Path(__file__).parent / 'checkpoints'

# for testing/debugging purposes to train models faster

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    return args


def train(model, dataloader, optimizer, train_GAN=True):
    pbar = tqdm(dataloader, total=len(dataloader))
    for i, (ciphertext, plaintext) in enumerate(pbar):
        
        with tf.GradientTape() as tape:
            probs = model(ciphertext[:, 1:])
            loss = model.loss_vanilla(probs, plaintext[:, 1:])
            if train_GAN:
                loss += model.loss_G(probs)
        if not train_GAN or i <= len(dataloader) // 2:
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = model.accuracy(probs, plaintext[:, 1:])
        
        probs = model(ciphertext[:, 1:])
        pred = tf.argmax(probs, axis=-1)
        with tf.GradientTape() as tape:
            loss_D = model.loss_D(pred, plaintext[:, 1:])
        if train_GAN and i > len(dataloader) // 2:
            gradients = tape.gradient(loss_D, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        pbar.set_description(f'Loss: {(loss.numpy().item() if loss else 0):.2f} Loss (D): {(loss_D.numpy().item() if loss_D else 0):.2f} Accuracy: {acc.item():.2f}')
        
    print(detokenize(plaintext[:, 1:][0]), "\t", detokenize(tf.argmax(input=probs, axis=2)[0]))
    return

def test(model, dataloader):
    print("TESTING ------------------")
    acc_list = []
    for i, (ciphertext, plaintext) in enumerate(dataloader):
        #if i % REDUCTION_FACTOR != 0: continue  # TODO: comment out during actual evaluation
        probs = model(ciphertext[:, 1:])
        acc = model.accuracy(probs, plaintext[:, 1:])
        acc_list.append(acc)
        if i % len(dataloader) // 10 == 0:
            print(detokenize(plaintext[:, 1:][0]), "\t", detokenize(tf.argmax(input=probs, axis=2)[0]))
    acc = np.mean(acc_list)
    print(f"Accuracy: {acc}")
    return acc

def get_model(args):
    return AdversarialTransformer(len(tokenizer ))

def main(args):
    model = get_model(args)

    # load in ciphers
    caesar_ciphers = np.arange(len(VOCAB))
    np.random.shuffle(caesar_ciphers)
    
    ciphers = caesar_ciphers[:len(caesar_ciphers)]
    print(f'Training ciphers: {ciphers}')
    train_dataloader = []
    test_dataloader = []
    for c in ciphers:
        tr, te = load_data(
            DATA_DIR / f'caesar_{c}.npz', args.window_size, args.batch_size, use_pct=1 / len(ciphers))
        train_dataloader += list(tr)
        test_dataloader += list(te)
    train_len = len(train_dataloader)
    test_len = len(test_dataloader)
    train_indices = tf.random.shuffle(tf.range(0, train_len))
    train_dataloader = tf.gather(train_dataloader, train_indices)
    test_indices = tf.random.shuffle(tf.range(0, test_len))
    test_dataloader = tf.gather(test_dataloader, test_indices)
    
    if args.load:
        # run the model once before loading weights to figure out shapes
        # required by keras
        model(train_dataloader[0][0][:,1:])
        model.load_weights(args.load)
        # test(model, test_dataloader)     
        print(model.decrypt('phjdq iulvhood pdvvlyh dev srs rii'))
        print(model.decrypt('7nkpdan fkdj paj ej8d 9e8g i6gao ia sap'))
        print(model.decrypt('6hat 9ejc snepejc lanba8p 8k9a 7a iu p6'))
        print(model.decrypt('ymj htrrzsnxyx inxifns yt htshjfq ymjnw 0nj1x fsi fnrx ymj3 tujsq3 ijhqfwj ymfy ymjnw jsix hfs gj fyyfnsji tsq3 g3 ymj ktwhngqj t0jwymwt1 tk fqq j2nxynsl xthnfq htsinyntsx qjy ymj wzqnsl hqfxxjx ywjrgqj fy f htrrzsnxynh wj0tqzynts ymj uwtqjyfwnfsx mf0j stymnsl yt qtxj gzy ymjnw hmfnsx ymj3 mf0j f 1twqi yt 1ns'))
    else: 
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        for e in range(args.num_epochs):
            print(f'EPOCH {e+1} of {args.num_epochs} ------------------')
            train(model, train_dataloader, optimizer, train_GAN=e < 10)
            checkpoint_path = CKPT_DIR / 'ADVERSARIAL_TRANSFORMER' / f'{e:04d}.ckpt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(checkpoint_path)
        acc = test(model, test_dataloader)
        
        
        
if __name__ == '__main__':
    args = parseArguments()
    main(args)
    # for i in range():
        
    #     loss_list = []
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    #     for e in range(args.num_epochs):
    #         print(f'EPOCH {e} of {args.num_epochs} ------------------')
    #         loss_list += train(model, train_dataloader, optimizer)

    #         #checkpoint_path = CKPT_DIR / 'caesar_rnn' / \
    #         #    f'{"+".join(map(lambda x: str(x), ciphers))}' / \
    #         #    f'{e:04d}.ckpt'
    #         #checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    #         #model.save_weights(checkpoint_path)

    #     # TODO: uncomment to display loss plots 
    #     #plt.plot(loss_list)
    #     #plt.show()
        
    #     test(model, test_dataloader)

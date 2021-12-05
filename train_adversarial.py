from utils.utils import DATA_DIR, tokenizer, detokenize
from utils.ciphers import VOCAB
import tensorflow as tf
import numpy as np
from models.RNN import RNN
from models.simple_RNN import Simple_RNN
from models.transformer import Transformer
from models.simple_transformer import Simple_Transformer
from models.beefy_transformer import Beefy_Transformer
from models.beefy_adversarial_transformer import BeefyAdversarialTransformer
from tqdm import tqdm
import argparse 
from dataloaders.caesar import CaesarDataset
from dataloaders.vigenere import VigenereDataset
from dataloaders.substitution import SubstitutionDataset
from pathlib import Path
import matplotlib.pyplot as plt

CKPT_DIR = Path(__file__).parent / 'checkpoints'


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--load", type=str, default=None)
    args = parser.parse_args()
    return args

def train(model, dataset, optimizer, train_GAN=True):
    dataloader = dataset.get_train_epoch()
    pbar = tqdm(dataloader, total=dataset.get_train_len())
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
        
        if i == dataset.get_train_len() - 1:
            print(detokenize(plaintext[0]), "\t", detokenize(
                tf.argmax(input=probs, axis=2)[0]))

def test(model, dataset, show_k=10):
    print("TESTING ------------------")
    dataloader = dataset.get_test_epoch()
    acc_list = []
    for i, (ciphertext, plaintext) in enumerate(dataloader):
        probs = model(ciphertext)
        acc = model.accuracy(probs, plaintext)
        acc_list.append(acc)
        if i < show_k:
            print(detokenize(plaintext[0]), "\t", detokenize(
                tf.argmax(input=probs, axis=2)[0]))
    acc = np.mean(acc_list)
    print(f"Accuracy: {acc}")
    return acc

def get_model(args):
    if args.model == 'SIMPLE_RNN':
        return Simple_RNN(len(tokenizer))
    elif args.model == 'SIMPLE_TRANSFORMER':
        return Simple_Transformer(len(tokenizer))
    elif args.model == 'TRANSFORMER':
        return Transformer(len(tokenizer))
    elif args.model == 'RNN':
        return RNN(len(tokenizer))
    elif args.model == 'BEEFY_TRANSFORMER':
        return Beefy_Transformer(len(tokenizer))
    elif args.model == 'BEEFY_ADVERSARIAL_TRANSFORMER':
        return BeefyAdversarialTransformer(len(tokenizer))


def get_dataset(args):
    if args.dataset == "CAESAR":
        return CaesarDataset(range(36), args.batch_size, args.window_size)
    elif args.dataset == "SUBSTITUTION":
        return SubstitutionDataset(args.batch_size, args.window_size)
    elif args.dataset == "VIGENERE":
        return VigenereDataset(args.batch_size, args.window_size)


def main(args):
    model = get_model(args)
    dataset = get_dataset(args)

    if args.load:
        # run the model once before loading weights to figure out shapes
        # required by keras
        model(next(dataset.get_train_epoch())[0][0][:, 1:])
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
            train(model, dataset, optimizer, train_GAN=e >= 25)
            checkpoint_path = CKPT_DIR / str(args.model) / f'{e:04d}.ckpt'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_weights(checkpoint_path)
        test(model, dataset)
        
        
        
if __name__ == '__main__':
    args = parseArguments()
    main(args)
    #     # TODO: uncomment to display loss plots
    #     #plt.plot(loss_list)
    #     #plt.show()

    #     test(model, test_dataloader)

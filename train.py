from utils.utils import get_batches
import tensorflow as tf
import numpy as np

def train(model, train_encoder, train_decoder, padding_index, epochs):
    for e in range(epochs):
        print(f"Epoch {e + 1}: ")

        for encoder, decoder in get_batches(model.batch_size, train_encoder, train_decoder):       
            with tf.GradientTape() as tape:
                probs = model(encoder, decoder)
                mask = np.where(decoder == padding_index, 0, 1)
                loss = model.loss_function(probs, decoder, mask)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

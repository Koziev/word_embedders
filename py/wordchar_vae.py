"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

import os
import io
import pickle

import platform
import colorama
import terminaltables
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split



FILLER_CHAR = ' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = '['  # символ отмечает начало цепочки символов слова
END_CHAR = ']'  # символ отмечает конец цепочки символов слова


"""
## Create a sampling layer
"""
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) * 0.5
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


"""
## Define the VAE as a `Model` with a custom `train_step`
"""
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    #keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=(1,)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def create_model(params, computed_params):
    latent_dim = params['latent_dim']
    char_dim = params['char_dim']
    max_len = computed_params['max_len']
    nb_chars = computed_params['nb_chars']

    """
    ## Build the encoder
    """
    encoder_inputs = keras.Input(shape=(max_len,), dtype='int32')
    #encoder_inputs = keras.Input(batch_input_shape=(params['batch_size'], max_len,), dtype='int32')

    emb = Embedding(input_dim=nb_chars,
                    output_dim=char_dim,
                    mask_zero=True,
                    trainable=True)(encoder_inputs)

    x = layers.LSTM(units=latent_dim, return_sequences=False)(emb)
    #x = layers.Dense(latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    """
    ## Build the decoder
    """
    #latent_inputs = keras.Input(batch_input_shape=(params['batch_size'], latent_dim,), batch_shape=params['batch_size'])
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = RepeatVector(max_len)(latent_inputs)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    decoder_outputs = TimeDistributed(layers.Dense(units=nb_chars, activation='softmax'), name='output')(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

    decoder.summary()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam())

    return vae


def print_red_line(msg):
    print(colorama.Fore.RED + msg + colorama.Fore.RESET)


def print_green_line(msg):
    print(colorama.Fore.GREEN + msg + colorama.Fore.RESET)


def get_ok_label():
    if platform.system() == 'Windows':
        return u'(+) '
    else:
        return u'☑ '


def get_fail_label():
    if platform.system() == 'Windows':
        return u'(-) '
    else:
        return u'☒ '


class VisualizeCallback(keras.callbacks.Callback):
    def __init__(self, X_viz, samples_viz, vae, computed_params, save_dir):
        self.epoch = 0
        self.save_dir = save_dir
        self.X = X_viz
        self.samples = samples_viz
        self.vae = vae
        self.index2char = dict((i, c) for c, i in computed_params['ctoi'].items())
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 5

    def decode_char_indeces(self, char_indeces):
        w = ''.join([self.index2char[c] for c in char_indeces]).strip()
        if w[0] == BEG_CHAR:
            w = w[1:]
        if w[-1] == END_CHAR:
            w = w[:-1]
        return w.strip()

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        nb_samples = 0
        nb_errors = 0

        z_mean, _, _ = self.vae.encoder(self.X)
        word_vs = z_mean.numpy()

        x_decoded = self.vae.decoder.predict(word_vs)
        x_decoded = np.argmax(x_decoded, axis=-1)

        viztable = ['accuracy true_word reconstructed_word'.split()]
        for i in np.random.permutation(np.arange(self.X.shape[0])):
            true_word = self.samples[i]
            pred_word = self.decode_char_indeces(x_decoded[i])

            if len(viztable) < 10:
                hit = true_word == pred_word
                if hit:
                    label = colorama.Fore.GREEN + get_ok_label() + colorama.Fore.RESET
                else:
                    label = colorama.Fore.RED + get_fail_label() + colorama.Fore.RESET
                viztable.append([label, true_word, pred_word])

            nb_samples += 1
            if true_word != pred_word:
                nb_errors += 1

        print(terminaltables.AsciiTable(viztable).table)

        val_acc = float(nb_samples - nb_errors) / nb_samples

        if val_acc > self.best_val_acc:
            print_green_line('\nInstance accuracy improved from {} to {}\n'.format(self.best_val_acc, val_acc))
            self.best_val_acc = val_acc
            self.vae.encoder.save_weights(os.path.join(self.save_dir, 'wordchar_vae.encoder.weights'))
            self.vae.decoder.save_weights(os.path.join(self.save_dir, 'wordchar_vae.decoder.weights'))
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve (current best acc={})\n'.format(val_acc, self.best_val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc


def decode_vector(decoder, v, itoc):
    z_sample = np.expand_dims(v, 0)
    x_decoded = decoder.predict(z_sample)
    x_decoded = np.argmax(x_decoded, axis=-1)
    chars = [itoc[i] for i in x_decoded[0]]
    if chars[0] == BEG_CHAR:
        chars = chars[1:]

    if END_CHAR in chars:
        chars = chars[:chars.index(END_CHAR)]

    s = ''.join(chars)
    return s


if __name__ == '__main__':
    tmp_dir = '../tmp'
    data_dir = '../data'
    save_dir = '../tmp/wordchar_vae'

    batch_size = 256

    params = dict()
    params['batch_size'] = batch_size

    computed_params = dict()

    dataset_path = os.path.join(data_dir, 'words.txt')

    with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
        words = [l.strip() for l in rdr]

    max_len = max(map(len, words)) + 2  # два граничных маркера
    computed_params['max_len'] = max_len

    all_chars = set()
    for word in words:
        all_chars.update(word)

    ctoi = dict((c, i) for i, c in enumerate(all_chars, start=3))
    ctoi[FILLER_CHAR] = 0
    ctoi[BEG_CHAR] = 1
    ctoi[END_CHAR] = 2

    X = np.zeros((len(words), max_len), dtype=np.int)
    for irow, word in enumerate(words):
        word2 = BEG_CHAR + word + END_CHAR
        for ichar, c in enumerate(word2):
            X[irow, ichar] = ctoi[c]

    X_train, X_viz, samples_train, samples_viz = train_test_split(X, words, test_size=10*batch_size)

    computed_params['nb_chars'] = max(ctoi.values()) + 1
    computed_params['ctoi'] = ctoi

    params['latent_dim'] = 64
    params['char_dim'] = 64

    vae = create_model(params, computed_params)

    visualizer = VisualizeCallback(X_viz, samples_viz, vae, computed_params, save_dir)

    print('Start training VAE on {} samples...'.format(X_train.shape[0]))
    vae.fit(X_train,
            epochs=100,
            verbose=2,
            batch_size=params['batch_size'],
            callbacks=[visualizer],
            )

    # Сохраняем части модели для инференса.
    with open(os.path.join(save_dir, 'wordchar_vae.encoder.arch'), 'w') as f:
        f.write(vae.encoder.to_json())

    with open(os.path.join(save_dir, 'wordchar_vae.decoder.arch'), 'w') as f:
        f.write(vae.decoder.to_json())

    with open(os.path.join(save_dir, 'wordchar_vae.config'), 'wb') as f:
        pickle.dump({**params, **computed_params}, f)

    exit(0)

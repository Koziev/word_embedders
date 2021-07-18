"""
Тренировка модели для превращения символьной цепочки слова в вектор действительных чисел фиксированной длины.
Кроме базовой автоэнкодерной архитектуры (см. реализацию в wordchar_simple_autoencoder.py), сетка имеет
дополнительные головы - классификатор части речи etc
"""

import numpy as np
import os
import io
import json
import logging
import platform
import glob

import absl.logging  # https://github.com/tensorflow/tensorflow/issues/26691
import colorama
import coloredlogs
import terminaltables
import pyconll

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.callbacks
from keras.layers.core import RepeatVector
from keras.layers import recurrent
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Input
from keras.layers import Conv1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.models import Model


FILLER_CHAR = ' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = '['  # символ отмечает начало цепочки символов слова
END_CHAR = ']'  # символ отмечает конец цепочки символов слова


class Sample:
    def __init__(self, word, lemma, part_of_speech):
        self.word = word
        self.lemma = lemma
        self.part_of_speech = part_of_speech

    def __repr__(self):
        return self.word + '(' + self.part_of_speech + ')'


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


def pad_word(word, max_word_len):
    return BEG_CHAR + word + END_CHAR + (max_word_len - len(word)) * FILLER_CHAR


def unpad_word(word):
    return word.strip()[1:-1]


def vectorize_sample(sample, X_batch, y_batch, y_pos_batch, irow, char2index, pos2index):
    y_pos_batch[irow, pos2index[sample.part_of_speech]] = 1

    for ich, ch in enumerate([BEG_CHAR]+list(sample.word)+[END_CHAR]):
        # TODO - для denoising AE будут разные символьные цепочки на входе и выходе!
        X_batch[irow, ich] = char2index[ch]
        y_batch[irow, ich] = char2index[ch]


def vectorize_data(samples, char2index, seq_len, pos2index):
    nb_samples = len(samples)
    X = np.zeros((nb_samples, seq_len), dtype=np.int32)
    y = np.zeros((nb_samples, seq_len), dtype=np.int32)
    y_pos = np.zeros((nb_samples, len(pos2index)), dtype=np.int32)

    for iword, sample in enumerate(samples):
        vectorize_sample(sample, X, y, y_pos, iword, char2index, pos2index)

    return X, y, y_pos


class SimpleAE(keras.Model):
    def __init__(self, encoder, decoder, latent_inputs, decoder_char_output, **kwargs):
        super(SimpleAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_inputs = latent_inputs
        self.decoder_char_output = decoder_char_output
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.part_of_speech_loss_tracker = keras.metrics.Mean(name="part_of_speech_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.part_of_speech_loss_tracker, self.total_loss_tracker]

    def train_step(self, data):
        # когда fit() получает тензоры x и y, наш аргумент data становится tuple(x_batch, y_batch)
        # поэтому далее используется data[0] и data[1]

        with tf.GradientTape() as tape:
            # прогоняем входные (возможно, это зашумленные слова) через энкодер
            z = self.encoder(data[0])

            # теперь выдачу энкодера прогоняем через декодер
            reconstruction, pred_pos = self.decoder(z)

            # вычисляем loss по разнице между эталонной выдачей и выходом декодера
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.sparse_categorical_crossentropy(data[1]['output_word'], reconstruction), axis=(1,)))
            pos_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data[1]['output_pos'], pred_pos)))
            total_loss = reconstruction_loss + pos_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.part_of_speech_loss_tracker.update_state(pos_loss)

        return {"loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "part_of_speech_loss": self.part_of_speech_loss_tracker.result()}

    def __call__(self, x, training=False):
        latent = self.encoder(x)
        return self.decoder(latent)

    def truncate_decoder(self):
        truncated_decoder = Model(inputs=self.latent_inputs, outputs=[self.decoder_char_output], name='decoder')
        return truncated_decoder


def create_model(arch_type, char_dims, tunable_char_embeddings, nb_chars, seq_len, pos_values, latent_dim):
    mask_zero = arch_type == 'rnn'

    if char_dims > 0:
        # Символы будут представляться векторами заданной длины,
        # и по мере обучения вектора будут корректироваться для
        # уменьшения общего лосса.
        embedding = Embedding(output_dim=char_dims,
                              input_dim=nb_chars,
                              input_length=seq_len,
                              mask_zero=mask_zero,
                              trainable=True)
    else:
        # 1-hot encoding of characters.
        # длина векторов пользователем не указана, поэтому задаем ее так, что
        # поместилось 1-hot представление.
        char_dims = nb_chars

        char_matrix = np.zeros((nb_chars, char_dims), dtype=np.float32)
        for i in range(nb_chars):
            char_matrix[i, i] = 1.0

        embedding = Embedding(output_dim=char_dims,
                              input_dim=nb_chars,
                              input_length=seq_len,
                              weights=[char_matrix],
                              mask_zero=mask_zero,
                              trainable=tunable_char_embeddings)

    encoder_inputs = Input(shape=(seq_len,), dtype='int32', name='input')
    encoder = embedding(encoder_inputs)

    if arch_type == 'cnn':
        conv_list = []
        merged_size = 0

        nb_filters = 32

        for kernel_size in range(1, 4):
            conv_layer = Conv1D(filters=nb_filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1)(encoder)
            # conv_layer = GlobalMaxPooling1D()(conv_layer)
            conv_layer = GlobalAveragePooling1D()(conv_layer)
            conv_list.append(conv_layer)
            merged_size += nb_filters

        encoder = keras.layers.concatenate(inputs=conv_list)
        encoder = Dense(units=latent_dim, activation='sigmoid')(encoder)

    elif arch_type == 'rnn':
        encoder = recurrent.LSTM(units=latent_dim, return_sequences=False)(encoder)

    elif arch_type == 'bidir_lstm':
        encoder = Bidirectional(recurrent.LSTM(units=latent_dim // 2, return_sequences=False))(encoder)

    elif arch_type == 'lstm(lstm)':
        encoder = Bidirectional(recurrent.LSTM(units=latent_dim // 2, return_sequences=True))(encoder)
        encoder = Bidirectional(recurrent.LSTM(units=latent_dim // 2, return_sequences=False))(encoder)

    elif arch_type == 'lstm+cnn':
        conv_list = []
        merged_size = 0

        rnn_size = latent_dim
        conv_list.append(recurrent.LSTM(units=rnn_size, return_sequences=False)(encoder))
        merged_size += rnn_size

        nb_filters = 32
        for kernel_size in range(1, 4):
            conv_layer = Conv1D(filters=nb_filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1)(encoder)
            # conv_layer = GlobalMaxPooling1D()(conv_layer)
            conv_layer = GlobalAveragePooling1D()(conv_layer)
            conv_list.append(conv_layer)
            merged_size += nb_filters

        encoder = keras.layers.concatenate(inputs=conv_list)
        encoder = Dense(units=latent_dim, activation='sigmoid')(encoder)

    elif arch_type == 'lstm(cnn)':
        conv_list = []
        merged_size = 0

        nb_filters = 32
        rnn_size = nb_filters

        for kernel_size in range(1, 4):
            conv_layer = Conv1D(filters=nb_filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1,
                                name='shared_conv_{}'.format(kernel_size))(encoder)

            # conv_layer = keras.layers.MaxPooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
            conv_layer = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
            conv_layer = recurrent.LSTM(rnn_size, return_sequences=False)(conv_layer)

            conv_list.append(conv_layer)
            merged_size += rnn_size

        encoder = keras.layers.concatenate(inputs=conv_list)
        encoder = Dense(units=latent_dim, activation='sigmoid')(encoder)

    elif arch_type == 'gru(cnn)':
        conv_list = []
        merged_size = 0

        for kernel_size, nb_filters in [(1, 16), (2, 32), (3, 64), (4, 128)]:
            conv_layer = Conv1D(filters=nb_filters,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1,
                                name='shared_conv_{}'.format(kernel_size))(encoder)

            conv_layer = keras.layers.AveragePooling1D(pool_size=kernel_size, strides=None, padding='valid')(conv_layer)
            conv_layer = recurrent.GRU(nb_filters, return_sequences=False)(conv_layer)

            conv_list.append(conv_layer)
            merged_size += nb_filters

        encoder = keras.layers.concatenate(inputs=conv_list)
        encoder = Dense(units=latent_dim, activation='sigmoid')(encoder)

    else:
        raise RuntimeError('Unknown architecture of neural net: {}'.format(arch_type))

    encoder = keras.Model(inputs=encoder_inputs, outputs=encoder, name="encoder")

    # =================== ДЕКОДЕР ====================

    latent_inputs = keras.Input(shape=(latent_dim,))

    # первая цель - исходное слово
    decoder_char_output = RepeatVector(seq_len)(latent_inputs)
    decoder_char_output = recurrent.LSTM(latent_dim, return_sequences=True)(decoder_char_output)
    decoder_char_output = TimeDistributed(Dense(nb_chars, activation='softmax'), name='output_word')(decoder_char_output)

    # вторая цель - метка части речи
    decoder_pos = Dense(len(pos_values), activation='softmax', name='output_pos')(latent_inputs)

    decoder = Model(inputs=latent_inputs, outputs=[decoder_char_output, decoder_pos], name='decoder')
    decoder.compile(loss={'output_word': 'sparse_categorical_crossentropy',
                          'output_pos': 'categorical_crossentropy'}, optimizer=keras.optimizers.Adam())
    decoder.summary(line_length=120)

    ae = SimpleAE(encoder, decoder, latent_inputs, decoder_char_output)
    ae.compile(optimizer=keras.optimizers.Adam())
    return ae


class colors:
    if platform.system() == 'Windows':
        ok = ''
        fail = ''
        close = ''
    else:
        ok = '\033[92m'
        fail = '\033[91m'
        close = '\033[0m'


class VisualizeCallback(keras.callbacks.Callback):
    """
    Класс занимается как визуализацией качества сетки в конце каждой эпохи обучения,
    так и выполняет функции EarlyStopping и ModelCheckpoint колбэков, контролируя
    per install accuracy для валидационного набора.
    """

    def __init__(self, X_test, y_test, y_pos_test, model, index2char, weights_path, learning_curve_filepath):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.y_pos_test = y_pos_test
        self.model = model
        self.index2char = index2char
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.weights_path = weights_path
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 20
        self.warmup_epochs = 50
        self.learning_curve_filepath = learning_curve_filepath
        with io.open(learning_curve_filepath, 'wt', encoding='utf-8') as wrt:
            wrt.write('epoch\tword_acc\tlength_acc\tfirst_char_acc\tsecond_char_acc\tlast_but_1_char_acc\tlast_char_acc\n')

    def decode_char_indeces(self, char_indeces):
        return ''.join([self.index2char[c] for c in char_indeces]).strip()

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        nb_samples = 0
        nb_errors = 0

        y_pred, _ = self.model.predict(self.X_test, verbose=0)

        viztable = ['accuracy true_word reconstructed_word'.split()]

        # расчет точности восстановления первого, второго, предпоследнего и последнего символа.
        first_char_hits = []
        second_char_hits = []
        last1_char_hits = []
        last_char_hits = []
        length_hits = []

        for i in np.random.permutation(np.arange(self.X_test.shape[0])):
            true_word = unpad_word(self.decode_char_indeces(self.y_test[i]))
            pred_word = unpad_word(self.decode_char_indeces(y_pred[i, :, :].argmax(axis=-1)))

            length_hits.append(len(true_word) == len(pred_word))
            if len(true_word) >= 4 and len(pred_word) >= 4:
                first_char_hits.append(true_word[0] == pred_word[0])
                second_char_hits.append(true_word[1] == pred_word[1])
                last1_char_hits.append(true_word[-2] == pred_word[-2])
                last_char_hits.append(true_word[-1] == pred_word[-1])

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

        # Всякие прикольные метрики, которые будем смотреть в динамике обучения - точность реконструкции длины,
        # точность первого символа, точность второго символа и т.д.
        length_acc = np.mean(length_hits)
        first_char_acc = np.mean(first_char_hits)
        second_char_acc = np.mean(second_char_hits)
        last1_char_acc = np.mean(last1_char_hits)
        last_char_acc = np.mean(last_char_hits)
        with io.open(self.learning_curve_filepath, 'at', encoding='utf-8') as wrt:
            wrt.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(self.epoch, val_acc, length_acc, first_char_acc, second_char_acc, last1_char_acc, last_char_acc))

        if val_acc > self.best_val_acc:
            print_green_line('\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path))
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve (current best acc={})\n'.format(val_acc, self.best_val_acc))
            if self.epoch > self.warmup_epochs:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = self.epoch
                    self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc

    def new_epochs(self):
        self.wait = 0
        self.model.stop_training = False


class Wordchar2Vector_Trainer(object):
    """
    Класс реализует обучение нейросетевой модели для кодирования слов.
    """
    def __init__(self, arch_type, tunable_char_embeddings, char_dims,
                 model_dir, vec_size, batch_size):
        self.arch_type = arch_type
        self.tunable_char_embeddings = tunable_char_embeddings
        self.char_dims = char_dims
        self.model_dir = model_dir
        self.vec_size = vec_size
        self.batch_size = batch_size
        self.config_filename = 'wordchar2vector.config'
        self.seed = 1234567

    def load_words(self, corpus_dir):
        keys = set()
        samples = []
        for p in glob.glob(corpus_dir+'/*.conllu', recursive=False):
            for parsing in pyconll.load_from_file(p):
                for token in parsing:
                    word = token.form.lower()
                    lemma = token.lemma
                    pos = token.upos
                    if pos is not None and BEG_CHAR not in word and END_CHAR not in word and FILLER_CHAR not in word:
                        k = (word, pos)
                        if k not in keys:
                            samples.append(Sample(word, lemma, pos))
                            keys.add(k)

        return samples

    def train(self, corpus_dir, save_dir):
        # составляем список слов для тренировки и валидации
        samples = self.load_words(corpus_dir)
        logging.info('There are %d samples', len(samples))

        max_word_len = max(len(s.word) for s in samples)
        seq_len = max_word_len + 2  # 2 символа добавляются к каждому слову для маркировки начала и конца последовательности
        logging.info('max_word_len=%d', max_word_len)

        # Метки части речи
        pos_values = list(set(s.part_of_speech for s in samples))
        pos2index = dict((p, i) for i, p in enumerate(pos_values))

        train_samples, val_samples = train_test_split(samples, test_size=1000)

        all_chars = {FILLER_CHAR, BEG_CHAR, END_CHAR}
        for s in samples:
            all_chars.update(s.word)

        char2index = dict((c, i) for i, c in enumerate([c for c in all_chars if c not in (FILLER_CHAR, BEG_CHAR, END_CHAR)], start=3))
        char2index[FILLER_CHAR] = 0
        char2index[BEG_CHAR] = 1
        char2index[END_CHAR] = 2

        index2char = dict([(i, c) for c, i in char2index.items()])

        nb_chars = max(char2index.values())+1
        logging.info('nb_chars=%d', nb_chars)

        model = create_model(arch_type, char_dims, tunable_char_embeddings, nb_chars, seq_len, pos_values, vec_size)

        model_config = {
            'max_word_len': max_word_len,
            'seq_len': seq_len,
            'char2index': char2index,
            'FILLER_CHAR': FILLER_CHAR,
            'BEG_CHAR': BEG_CHAR,
            'END_CHAR': END_CHAR,
            'vec_size': self.vec_size,
            'arch_type': self.arch_type,
            'pos2index': pos2index
        }

        with open(os.path.join(self.model_dir, self.config_filename), 'w') as f:
            json.dump(model_config, f, indent=4)

        weigths_path = os.path.join(self.model_dir, 'multitarget_simple_autoencoder.tmp.model')

        X_train, y_train, y_pos_train = vectorize_data(train_samples, char2index, seq_len, pos2index)
        X_val, y_val, y_pos_val = vectorize_data(val_samples, char2index, seq_len, pos2index)

        learning_curve_path = os.path.join(save_dir, 'learning_curve.tsv')
        visualizer = VisualizeCallback(X_val, y_val, y_pos_val, model, index2char, weigths_path, learning_curve_path)

        batch_size = self.batch_size
        logging.info('Start training with batch_size=%d', batch_size)
        visualizer.new_epochs()
        hist = model.fit(x=X_train, y={'output_word': y_train, 'output_pos': y_pos_train},
                         batch_size=batch_size,
                         epochs=1000,
                         verbose=1,
                         shuffle=True,
                         callbacks=[visualizer],
                         validation_data=(X_val, {'output_1': y_val, 'output_2': y_pos_val}),
                        )
        logging.info('Training complete, best_accuracy={}'.format(visualizer.get_best_accuracy()))

        # Загружаем наилучшее состояние модели
        model.load_weights(weigths_path)
        #os.remove(weigths_path)

        # Сохраним энкодерную часть
        with open(os.path.join(self.model_dir, 'wordchar2vector.encoder.arch'), 'w') as f:
            f.write(model.encoder.to_json())

        model.encoder.save_weights(os.path.join(self.model_dir, 'wordchar2vector.encoder.model'))

        # У декодера надо оставить только первую голову, а вспомогательные цели откусить
        decoder = model.truncate_decoder()

        with open(os.path.join(self.model_dir, 'wordchar2vector.decoder.arch'), 'w') as f:
            f.write(decoder.to_json())

        decoder.save_weights(os.path.join(self.model_dir, 'wordchar2vector.decoder.model'))


def init_trainer_logging(logfile_path, debugging=True):
    # настраиваем логирование в файл и эхо-печать в консоль

    # https://github.com/tensorflow/tensorflow/issues/26691
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    log_level = logging.DEBUG if debugging else logging.ERROR
    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger('')
    logger.setLevel(log_level)

    file_fmt = '%(asctime)-15s %(levelname)-7s %(name)-25s %(message)s'

    if logfile_path:
        lf = logging.FileHandler(logfile_path, mode='w')
        lf.setLevel(logging.DEBUG)
        formatter = logging.Formatter(file_fmt)
        lf.setFormatter(formatter)
        logging.getLogger('').addHandler(lf)

    if True:
        field_styles = coloredlogs.DEFAULT_FIELD_STYLES.copy()
        field_styles["asctime"] = {}
        level_styles = coloredlogs.DEFAULT_LEVEL_STYLES.copy()
        level_styles["debug"] = {}
        coloredlogs.install(
            level=log_level,
            use_chroot=False,
            fmt=file_fmt,  #"%(asctime)s %(levelname)-8s %(name)s  - %(message)s",
            level_styles=level_styles,
            field_styles=field_styles,
        )


if __name__ == '__main__':
    data_dir = '../data'
    tmp_dir = '../tmp'
    model_dir = os.path.join(tmp_dir, 'multitarget_simple_autoencoder')

    vec_size = 56
    char_dims = 0
    tunable_char_embeddings = False
    arch_type = 'rnn'  # choices = cnn | rnn | bidir_lstm | lstm(lstm) | lstm+cnn | lstm(cnn) | gru(cnn)
    batch_size = 350

    # настраиваем логирование в файл
    init_trainer_logging(os.path.join(tmp_dir, 'multitarget_simple_autoencoder.log'))

    trainer = Wordchar2Vector_Trainer(arch_type,
                                      tunable_char_embeddings,
                                      char_dims,
                                      model_dir,
                                      vec_size,
                                      batch_size)

    trainer.train(os.path.join(data_dir, "SynTagRus"), model_dir)

    logging.info('Done.')

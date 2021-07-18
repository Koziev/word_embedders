"""
Тренировка модели для превращения символьной цепочки слова в вектор действительных чисел фиксированной длины.

Реализации RNN и CNN вариантов энкодера, включая комбинации. Реализовано на Keras.
Подробности: https://github.com/Koziev/chatbot/blob/master/ruchatbot/trainers/README.wordchar2vector.md
"""

import numpy as np
import os
import io
import json
import logging
import platform

import absl.logging  # https://github.com/tensorflow/tensorflow/issues/26691
import colorama
import coloredlogs
import terminaltables

from sklearn.model_selection import train_test_split
import sklearn.metrics
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


def raw_wordset(wordset, max_word_len):
    return [(pad_word(word, max_word_len), pad_word(word, max_word_len)) for word in wordset]


def vectorize_word(word, corrupt_word, X_batch, y_batch, irow, char2index):
    for ich, (ch, corrupt_ch) in enumerate(zip(word, corrupt_word)):
        if corrupt_ch not in char2index:
            print('Char "{}" code={} word="{}" missing in char2index'.format(corrupt_ch, ord(corrupt_ch), corrupt_word))
        else:
            X_batch[irow, ich] = char2index[corrupt_ch]

        if ch not in char2index:
            print('Char "{}" code={} word="{}" missing in char2index'.format(ch, ord(ch), word))
        else:
            y_batch[irow, ich] = char2index[ch]


def vectorize_data(wordset, char2index, seq_len):
    X = np.zeros((len(wordset), seq_len), dtype=np.int32)
    y = np.zeros((len(wordset), seq_len), dtype=np.int32)

    for iword, (word, corrupt_word) in enumerate(wordset):
        vectorize_word(word, corrupt_word, X, y, iword, char2index)

    return X, y


class SimpleAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(SimpleAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        # когда fit() получает тензоры x и y, наш аргумент data становится tuple(x_batch, y_batch)
        # поэтому далее используется data[0] и data[1]

        with tf.GradientTape() as tape:
            # прогоняем входные (возможно, это зашумленные слова) через энкодер
            z = self.encoder(data[0])

            # теперь выдачу энкодера прогоняем через декодер
            reconstruction = self.decoder(z)

            # вычисляем loss по разнице между эталонной выдачей и выходом декодера
            total_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.sparse_categorical_crossentropy(data[1], reconstruction), axis=(1,))
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)

        return {"loss": self.total_loss_tracker.result(),}

    def __call__(self, x, training=False):
        latent = self.encoder(x)
        return self.decoder(latent)


def create_model(arch_type, char_dims, tunable_char_embeddings, nb_chars, seq_len, latent_dim):
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

    decoder_outputs = RepeatVector(seq_len)(latent_inputs)
    decoder_outputs = recurrent.LSTM(latent_dim, return_sequences=True)(decoder_outputs)
    decoder_outputs = TimeDistributed(Dense(nb_chars, activation='softmax'), name='output')(decoder_outputs)

    decoder = Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')
    decoder.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

    ae = SimpleAE(encoder, decoder)
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

    def __init__(self, X_test, y_test, model, index2char, weights_path, learning_curve_filename):
        self.epoch = 0
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.index2char = index2char
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.weights_path = weights_path
        self.learning_curve_filename = learning_curve_filename
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 20
        with io.open(learning_curve_filename, 'wt', encoding='utf-8') as wrt:
            wrt.write('epoch\tword_acc\tlength_acc\tfirst_char_acc\tsecond_char_acc\tlast_but_1_char_acc\tlast_char_acc\n')

    def decode_char_indeces(self, char_indeces):
        return ''.join([self.index2char[c] for c in char_indeces]).strip()

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        nb_samples = 0
        nb_errors = 0

        y_pred = self.model.predict(self.X_test, verbose=0)

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
        with io.open(self.learning_curve_filename, 'at', encoding='utf-8') as wrt:
            wrt.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(self.epoch, val_acc, length_acc, first_char_acc, second_char_acc, last1_char_acc, last_char_acc))

        if val_acc > self.best_val_acc:
            print_green_line('\nInstance accuracy improved from {} to {}, saving model to {}\n'.format(self.best_val_acc, val_acc, self.weights_path))
            self.best_val_acc = val_acc
            self.model.save_weights(self.weights_path)
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

    def load_words(self, words_filepath):
        # из указанного текстового файла загружаем список слов без повторов
        # и возвращаем его для тренировки или векторизации.
        with io.open(words_filepath, 'r', encoding='utf-8') as rdr:
            return set([line.strip() for line in rdr])

    def train(self, words_filepath, save_dir):
        """
        Тренируем модель на словах в указанном файле words_filepath.

        :param words_filepath: путь к plain text utf8 файлу со списком слов (одно слово на строку)
        :param tmp_dir: путь к каталогу, куда будем сохранять всякие сводки по процессу обучения
        для визуализации и прочего контроля
        """

        # составляем список слов для тренировки и валидации
        known_words = self.load_words(words_filepath)
        logging.info('There are %d known words', len(known_words))

        max_word_len = max(map(len, known_words))
        seq_len = max_word_len + 2  # 2 символа добавляются к каждому слову для маркировки начала и конца последовательности
        logging.info('max_word_len=%d', max_word_len)

        train_words, val_words = train_test_split(list(known_words), test_size=1000)

        # В тренировочный набор добавляем особое "пустое" слово, которое нужно
        # в качестве заполнителя при выравнивании цепочек слов разной длины.
        train_words.append('')

        train_words = raw_wordset(train_words, max_word_len)
        val_words = raw_wordset(val_words, max_word_len)

        logging.info('train set contains %d words', len(train_words))
        logging.info('val set contains %d words', len(val_words))

        all_chars = {FILLER_CHAR, BEG_CHAR, END_CHAR}
        for word in known_words:
            all_chars.update(word)

        char2index = {FILLER_CHAR: 0}
        for i, c in enumerate(all_chars):
            if c != FILLER_CHAR:
                char2index[c] = len(char2index)

        index2char = dict([(i, c) for c, i in char2index.items()])

        nb_chars = len(all_chars)
        logging.info('nb_chars=%d', nb_chars)

        model = create_model(arch_type, char_dims, tunable_char_embeddings, nb_chars, seq_len, vec_size)

        model_config = {
            'max_word_len': max_word_len,
            'seq_len': seq_len,
            'char2index': char2index,
            'FILLER_CHAR': FILLER_CHAR,
            'BEG_CHAR': BEG_CHAR,
            'END_CHAR': END_CHAR,
            'vec_size': self.vec_size,
            'arch_type': self.arch_type
        }

        with open(os.path.join(self.model_dir, self.config_filename), 'w') as f:
            json.dump(model_config, f, indent=4)

        weigths_path = os.path.join(self.model_dir, 'wordchar2vector.tmp.model')

        X_train, y_train = vectorize_data(train_words, char2index, seq_len)
        X_val, y_val = vectorize_data(val_words, char2index, seq_len)

        learning_curve_filename = os.path.join(save_dir, 'learning_curve.tsv')
        visualizer = VisualizeCallback(X_val, y_val, model, index2char, weigths_path, learning_curve_filename)

        batch_size = self.batch_size
        logging.info('Start training with batch_size=%d', batch_size)
        visualizer.new_epochs()
        hist = model.fit(x=X_train, y=y_train,
                         batch_size=batch_size,
                         epochs=1000,
                         verbose=1,
                         callbacks=[visualizer],  # csv_logger, model_checkpoint, early_stopping],
                         validation_data=(X_val, y_val),
                        )
        logging.info('Training complete, best_accuracy={}'.format(visualizer.get_best_accuracy()))

        # Загружаем наилучшее состояние модели
        model.load_weights(weigths_path)
        #os.remove(weigths_path)

        # Сохраним энкодерную часть
        with open(os.path.join(self.model_dir, 'wordchar2vector.encoder.arch'), 'w') as f:
            f.write(model.encoder.to_json())

        model.encoder.save_weights(os.path.join(self.model_dir, 'wordchar2vector.encoder.model'))

        with open(os.path.join(self.model_dir, 'wordchar2vector.decoder.arch'), 'w') as f:
            f.write(model.decoder.to_json())

        model.decoder.save_weights(os.path.join(self.model_dir, 'wordchar2vector.decoder.model'))



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
    input_path = os.path.join(data_dir, 'words.txt')
    tmp_dir = '../tmp'
    model_dir = os.path.join(tmp_dir, 'simple_autoencoder')

    vec_size = 56
    char_dims = 0
    tunable_char_embeddings = False
    arch_type = 'rnn'  # choices = cnn | rnn | bidir_lstm | lstm(lstm) | lstm+cnn | lstm(cnn) | gru(cnn)
    batch_size = 350

    # настраиваем логирование в файл
    init_trainer_logging(os.path.join(tmp_dir, 'wordchar_simple_autoencoder.log'))

    trainer = Wordchar2Vector_Trainer(arch_type,
                                      tunable_char_embeddings,
                                      char_dims,
                                      model_dir,
                                      vec_size,
                                      batch_size)

    trainer.train(input_path, model_dir)

    logging.info('Done.')

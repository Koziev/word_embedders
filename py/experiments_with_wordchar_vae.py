"""
Эксперименты с моделью вариационного автоэнкодера для символьного представления слов.
Предполагается, что модель уже обучена с помощью wordchar_vae.py и ее файлы лежат в ../tmp/wordchar_vae
"""

import os
import io
import pickle
import re
import random
import collections

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sklearn
import tqdm
import numpy as np
import keras
from tensorflow.keras.models import model_from_json

from wordchar_vae import Sampling


FILLER_CHAR = ' '  # символ для выравнивания слов по одинаковой длине
BEG_CHAR = '['  # символ отмечает начало цепочки символов слова
END_CHAR = ']'  # символ отмечает конец цепочки символов слова


class WordCharEmbedder:
    """ Класс для работы с обученной моделью - векторизация слов, декодирование векторов. """
    def __init__(self):
        self.config = dict()
        self.encoder = None
        self.decoder = None

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'wordchar_vae.config'), 'rb') as f:
            self.config = pickle.load(f)

        self.ctoi = self.config['ctoi']
        self.itoc = dict((i, c) for c, i in self.ctoi.items())

        with open(os.path.join(model_dir, 'wordchar_vae.encoder.arch'), 'rb') as f:
            self.encoder = model_from_json(f.read(), custom_objects={'Sampling': Sampling})

        self.encoder.load_weights(os.path.join(model_dir, 'wordchar_vae.encoder.weights'))

        with open(os.path.join(model_dir, 'wordchar_vae.decoder.arch'), 'rb') as f:
            self.decoder = model_from_json(f.read())

        self.decoder.load_weights(os.path.join(model_dir, 'wordchar_vae.decoder.weights'))

    def encode_word(self, word):
        """ Преобразование единственного слова в вектор """
        x = np.zeros((1, self.config['max_len']))
        for i, c in enumerate([BEG_CHAR] + list(word) + [END_CHAR]):
            x[0, i] = self.ctoi[c]

        latent, _, _ = self.encoder(x)
        return latent[0].numpy()

    def encode_words(self, words, show_progress=True):
        """ Преобразование набора слов в векторы """
        n = len(words)
        batch_size = 100
        x = np.zeros((batch_size, self.config['max_len']))
        i0 = 0
        latents = []

        if show_progress:
            pbar = tqdm.tqdm(total=n)

        while i0 < n:
            i1 = min(n, i0 + batch_size)
            bs = i1 - i0

            x.fill(0)
            for iword, word in enumerate(words[i0:i1]):
                for i, c in enumerate([BEG_CHAR] + list(word) + [END_CHAR]):
                    x[iword, i] = self.ctoi.get(c, 0)

            latent, _, _ = self.encoder(x)
            latents.append(latent[:bs, :].numpy())
            i0 += batch_size
            if show_progress:
                pbar.update(batch_size)

        if show_progress:
            pbar.close()

        return np.vstack(latents)

    def decode_vector(self, word_vector):
        """ Для заданного вектора восстанавливаем символьное представление слова """
        x = np.copy(word_vector)
        x = np.expand_dims(x, 0)
        y = self.decoder(x)
        y = np.argmax(y, axis=-1)
        cx = []
        for yi in y[0]:
            cx.append(self.itoc[yi])

        if cx[0] == BEG_CHAR:
            cx = cx[1:]
        if END_CHAR in cx:
            cx = cx[:cx.index(END_CHAR)]

        cx = ''.join(cx).strip()
        return cx


def op3(embedder, word1, word2, word3):
    """
    Выполняем операцию word1+(word2-word3) над векторами слов и декодируем
    получившийся вектор в слово, которое будет напечатано.
    """
    v = embedder.encode_word(word1) + (embedder.encode_word(word2) - embedder.encode_word(word3))
    w = embedder.decode_vector(v)
    print('{} + ({} - {}) => {}'.format(word1, word2, word3, w))


tok_regex = re.compile(r'[%s\s]+' % re.escape('[  +<>`~; .,?？!-…№”“„{}/\'"–—:«»*]()）》\t'))
def tokenize(text):
    return [word for word in tok_regex.split(text) if len(word) > 0]


if __name__ == '__main__':
    tmp_dir = '../tmp'
    data_dir = '../data'
    model_dir = '../tmp/wordchar_vae'

    we = WordCharEmbedder()
    we.load(model_dir)

    print('\nInterpolation:')
    for word1, word2 in [('муха', 'слон'), ('работать', 'бездельничать'), ('земной', 'космический')]:
        v1 = we.encode_word(word1)
        v2 = we.encode_word(word2)
        steps = []
        last_s = ''
        for k in np.linspace(0.0, 1.0, 10):
            v = (1.0-k)*v1 + k*v2
            s = we.decode_vector(v)
            if s != last_s:
                last_s = s
                steps.append(s)
        print('{}'.format(' => '.join(steps)))

    print('\nWord vector arithmetics:')
    op3(we, 'кошка', 'собакой', 'собака')
    op3(we, 'мама', 'папы', 'папа')
    op3(we, 'голод', 'холодать', 'холод')

    # ========================================================================
    word2count = collections.Counter()
    with io.open(os.path.join(data_dir, 'sents.txt'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            for word in tokenize(line.strip()):
                if all((c in we.ctoi) for c in word):
                    word2count[word.lower()] += 1

    #common_words = [w for w, _ in word2count.most_common(1000)]
    common_words = random.sample(word2count.keys(), 1000)
    vectors = we.encode_words(common_words, show_progress=False)

    tsne = TSNE(n_components=2, random_state=123456, perplexity=100, n_iter=1000)
    tsne_xy = tsne.fit_transform(vectors)

    x = tsne_xy[:, 0]
    y = tsne_xy[:, 1]

    plt.figure(figsize=(50, 50))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], s=len(common_words[i])*100)
        plt.annotate(common_words[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(os.path.join(tmp_dir, 'vae.tsne.png'))

    # ========================================================================
    print('\nNearest words:')
    with io.open(os.path.join(data_dir, 'words.txt'), 'r', encoding='utf-8') as rdr:
        words = [l.strip() for l in rdr if len(l) < 25]
        words = random.sample(words, 100000)  # векторизация полного списка займет до часа времени, поэтому возьмем небольшое подмножество

    vectors = we.encode_words(words)

    for probe_word in '123 муха я голограмма среднегодовой крошечная стоять прокумекав'.split():
        v = we.encode_word(probe_word)
        cx = sklearn.metrics.pairwise.cosine_similarity(vectors, [v])
        word_cosines = [(w, cx[i, 0]) for i, w in enumerate(words) if w != probe_word]
        word_cosines = sorted(word_cosines, key=lambda z: -z[1])
        s = ', '.join(['{}({:5.3f})'.format(word, cos) for word, cos in word_cosines[:10]])
        print('{} => {}'.format(probe_word, s))

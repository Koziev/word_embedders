import io
import os
import random
import collections
import logging

import networkx as nx
from gensim.models import word2vec
import matplotlib.pyplot as plt


def decode_pos(pos):
    if pos in ('ДЕЕПРИЧАСТИЕ', 'ГЛАГОЛ', 'ИНФИНИТИВ'):
        return 'VERB'
    elif pos == 'СУЩЕСТВИТЕЛЬНОЕ':
        return 'NOUN'
    elif pos == 'ПРИЛАГАТЕЛЬНОЕ':
        return 'ADJ'
    elif pos == 'НАРЕЧИЕ':
        return 'ADV'
    elif pos == 'ПРЕДЛОГ':
        return 'ADP'
    else:
        return pos


if __name__ == '__main__':
    data_dir = '../data'
    tmp_dir = '../tmp'

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    word2links = collections.defaultdict(list)
    all_words = set()
    G = nx.Graph()

    with io.open(os.path.join(data_dir, 'links.csv'), 'r', encoding='utf-8') as rdr:
        for line in rdr:
            tx = line.strip().split('\t')
            if len(tx) == 5:
                word1 = tx[0].replace(' - ', '-').lower().replace('ё', 'е')
                pos1 = decode_pos(tx[1])
                word2 = tx[2].replace(' - ', '-').lower().replace('ё', 'е')
                pos2 = decode_pos(tx[3])
                relat = tx[4]

                #if relat in ('в_класс', 'член_класса'):
                #    continue

                k1 = word1 + '_' + pos1
                k2 = word2 + '_' + pos2

                all_words.add(k1)
                all_words.add(k2)

                if k1 != k2:
                    word2links[k1].append((k2, relat))
                    G.add_edge(k1, k2)

    # Визуализируем небольшой участок графа
    G_cat = nx.generators.ego_graph(G, 'программа_NOUN', radius=2)
    plt.figure(figsize=(15, 15))
    nx.draw(G_cat, with_labels=True)
    plt.savefig(os.path.join(tmp_dir, 'thesaurus_embeddings.png'))


    all_words = list(all_words)
    print('Building dataset...')
    samples = []
    while len(samples) < 5000000:
        cur_node = random.choice(all_words)
        chain = [cur_node]
        while len(chain) < 20:
            next_nodes = word2links.get(cur_node, [])
            if next_nodes:
                cur_node = random.choice(next_nodes)[0]
                chain.append(cur_node)
            else:
                break

        # Запишем во временный файл получившуюся цепочку.
        samples.append(chain)

    print('Start training word2vec model using {} samples'.format(len(samples)))
    model = word2vec.Word2Vec(samples,
                              size=64,
                              window=2,
                              cbow_mean=0,
                              min_count=1,
                              workers=4,
                              sorted_vocab=1,
                              iter=1)

    model.init_sims(replace=True)

    # Поиск ближайших слов
    for probe_word in 'кошка_NOUN сухость_NOUN бодрствовать_VERB золотистый_ADJ'.split():
        nx = []
        for i, p in enumerate(model.wv.most_similar(positive=probe_word, topn=5)):
            nx.append((p[0], p[1]))

        s = ', '.join(['{}({:5.3f})'.format(word, sim) for word, sim in nx])
        print('{} => {}'.format(probe_word, s))


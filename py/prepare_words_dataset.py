import io
import glob
import pyconll


words = set()

#with io.open('/home/inkoziev/polygon/chatbot/tmp/known_words.txt', 'r', encoding='utf-8') as rdr:
#    words.update(l.strip().lower() for l in rdr)

#with io.open('/home/inkoziev/polygon/text_generator/tmp/stress_model/dataset.tsv', 'r', encoding='utf-8') as rdr:
#    rdr.readline()
#    words.update(line.split('\t')[0].lower() for line in rdr if '’' not in line)


for p in glob.glob('../data/SynTagRus/*.conllu', recursive=False):
    for parsing in pyconll.load_from_file(p):
        for token in parsing:
            word = token.form.lower()
            if not any(c in word for c in ' []'):
                words.add(word)


print('Storing {} words'.format(len(words)))
with io.open('../data/words.txt', 'w', encoding='utf-8') as wrt:
    for word in words:
        wrt.write('{}\n'.format(word))

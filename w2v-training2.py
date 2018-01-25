from tokenizer import tokenize
from gensim.models import Word2Vec
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = []

class MySentences():
    def __iter__(self):
        for line in open(os.path.join("text", "full-tokenized.txt")):
            yield line.split()
'''
with open('text/full.txt') as f:
    for line in f:
        sentences.append(tokenize(line.rstrip('\n')))
'''

sentences = MySentences()
model = Word2Vec(sentences, size=100, alpha=0.065)

model.save('model-full-tokenized.bin')

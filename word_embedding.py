from gensim.models import Word2Vec
from tokenizer import tokenize
import numpy as np

model = Word2Vec.load('model-full-tokenized.bin')
def wordEmbedding(document):
    vec = []
    words = tokenize(document)
    for i in xrange(len(words)):
        vec.append(model[words[i]])
    return vec

def w2vAverage(document):
    words = tokenize(document)
    vec = np.zeros(100)
    sum = 0
    for i in xrange(len(words)):
        if words[i] in model:
            vec +=  model[words[i]]
            sum += 1
    return vec
    
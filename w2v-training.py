import csv
from tokenizer import tokenize
from gensim.models import Word2Vec

sentences = []

with open('text/training2000.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        sentences.append(tokenize(row[0]))
print sentences[0:10]
model = Word2Vec(sentences, size=80, alpha=0.065, iter=1000)
#print model.wv.most_similar(positive=['dengan'])
#model.save('model5000.bin')

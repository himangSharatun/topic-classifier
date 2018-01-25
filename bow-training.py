from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenizeSentence as tokenize
import csv
import json

sentences = []

with open('text/training2000.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        sentences.append(tokenize(row[0]))

print sentences[0:5]

# corpus = [
# 'melakukan penyemainan benih cabe kriting di dlm ruangan tumbuh sprti toge ternyata ini dampak ya',
# 'Mohon informasi  untuk potensi pasar sayur okra gmn bos? Trimakasih',
# 'Biarpun panen tp hargx lg gk bersahabat',
# 'Master mau tanya  cara menmbuat pupuk cair dari feses kelinci gimana yah'
# ]
 
vectorizer = CountVectorizer()
vectorizer.fit_transform(sentences).todense() 

with open ('vocabulary.json', 'w') as vocabFile:
    json.dump(vectorizer.vocabulary_ , vocabFile)

# print vectorizer.transform(["Mau tanya pak  pupuk apa ya buat memperbesar bunga nya."]).toarray()
# print vectorizer

from sklearn.feature_extraction.text import CountVectorizer
import json

vocabulary = json.load(open('vocabulary.json'))
vectorizer = CountVectorizer(vocabulary=vocabulary)

def tobow(string):
    return vectorizer.transform([string]).toarray()
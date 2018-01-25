from gensim.models import Word2Vec

model = Word2Vec.load('model-full.bin')
print(model)
print model.wv.most_similar(positive=['dengan'])
print model.wv.most_similar(positive=['pupuk'])
print model.wv.most_similar(positive=['jika'])
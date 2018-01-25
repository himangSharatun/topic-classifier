import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from word_embedding import w2vAverage

numpy.random.seed(7)

# # load dataset
# dataframe = pandas.read_csv("text/data-training.2.csv", header=None)
# dataset = dataframe.values
# X = dataset[:,0]
# Y = dataset[:,1]

X_dataframe = pandas.read_csv("text/data-only.csv", header=None)
X = X_dataframe.values
Y_dataframe = pandas.read_csv("text/label-only.csv", header=None)
Y = Y_dataframe.values

dummy_x = []
for text in X:
	dummy_x.append(numpy.array(w2vAverage(text[0])))
	


dummy_x = numpy.array(dummy_x)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
numpy.save('classifier/encoder.npy',encoder.classes_)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_shape=(100,), activation='relu'))
	model.add(Dense(150, activation='sigmoid'))
	model.add(Dense(8, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

model = baseline_model()

#training process
model.fit(dummy_x, dummy_y, epochs=100)

#Save to Json
model_json = model.to_json()
with open("classifier/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("classifier/model.h5")
print "Model has been saved"

# test = w2vAverage("Bagaimana cara budidaya hidroponik yang baik tapi dengan biaya yang murah dan efisien #Trims")
# print test
# # evaluate the model
# # scores = model.evaluate(dummy_x, dummy_y)
# # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# # #print encoder.inverse_transform(0) #bibit
# # #print encoder.inverse_transform(1) #hama
# # #print encoder.inverse_transform(2) #harga
# # #print encoder.inverse_transform(3) #other
# # #print encoder.inverse_transform(4) #penyakit
# # #print encoder.inverse_transform(5) #pertanian
# # #print encoder.inverse_transform(6) #pestisida
# # #print encoder.inverse_transform(7) #pupuk

# predict = model.predict(numpy.array([test]))
# print predict
# print encoder.inverse_transform(numpy.argmax(predict[0]))
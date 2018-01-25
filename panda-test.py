from keras.models import model_from_json
from word_embedding import w2vAverage
from sklearn.preprocessing import LabelEncoder
import numpy

#Load from json
json_file = open("classifier/model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("classifier/model.h5")

print "Model has been loaded"

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
test = w2vAverage("Ini kenapa ya, bagaimana cara menanggulanginnya?")

predict = loaded_model.predict(numpy.array([test]))
encoder = LabelEncoder()
encoder.classes_ = numpy.load('classifier/encoder.npy')
index = numpy.argmax(predict[0])
print index
print encoder.classes_[index]
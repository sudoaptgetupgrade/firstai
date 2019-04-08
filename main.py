from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

#Input the amount of training needed
epochs = input("How many epochs??: ")
batchsize = input("How big of a batch size??: ")
epochs = int(epochs)
batchsize = int(batchsize)

#load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.data.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#Create the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model.fit(X, Y, epochs=epochs, batch_size=batchsize)

#Evaluation
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
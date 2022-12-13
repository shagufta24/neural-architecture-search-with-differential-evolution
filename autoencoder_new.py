import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,Reshape, LeakyReLU as LR, Activation, Dropout, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import *
import numpy as np
from matplotlib import pyplot as plt

X_train = np.genfromtxt("x_train.csv",delimiter=",", dtype=float)
y_train = np.genfromtxt("y_train.csv",delimiter=",", dtype=float)
X_test = np.genfromtxt("x_test.csv",delimiter=",", dtype=float)
y_test = np.genfromtxt("y_test.csv",delimiter=",", dtype=float)

candidate = [16, 12, 10, 8, 8, 10, 12, 16, 6]

latent_size = candidate[-1]
input_layers = candidate[:4]
output_layers = candidate[4:-1]
output_size = 13
input_size = (13,)

encoder = Sequential()
encoder.add(InputLayer(input_shape=input_size))
for nodes in input_layers:
    encoder.add(Dense(nodes, activation='tanh'))
encoder.add(Dense(latent_size, activation='tanh'))

decoder = Sequential()
decoder.add(InputLayer(input_shape=(latent_size,)))
for nodes in output_layers:
    decoder.add(Dense(nodes, activation='tanh'))
decoder.add(Dense(output_size, activation='tanh'))

inp = Input(shape = input_size)
latent_vector = encoder(inp)
output = decoder(latent_vector)
model = Model(inputs = inp, outputs = output)
model.compile("nadam", metrics= [keras.metrics.Accuracy()], loss = MeanAbsolutePercentageError())

history = model.fit(X_train, X_train, epochs=200, batch_size=64, verbose=1, validation_data=(X_test,X_test))

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# summarize history for loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
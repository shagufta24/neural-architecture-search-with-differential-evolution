import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input, Flatten,\
                                    Reshape, LeakyReLU as LR,\
                                    Activation, Dropout, InputLayer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import *
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Read data
df = pd.read_csv('data.csv')
df.drop(df.columns[[0]], axis=1, inplace=True)

# Split data
X = df.iloc[:, :-1]
X = X.to_numpy()
y = df.iloc[:, -1]
y = y.to_numpy()
X_train, X_val, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

output_size = X.shape[1]
input_size = (X.shape[1],)

def train_model(candidate):
    latent_size = candidate[-1]
    input_layers = candidate[:4]
    output_layers = candidate[4:-1]

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

    inputs = Input(shape = input_size)
    latent_vector = encoder(inputs)
    output = decoder(latent_vector)

    model = Model(inputs = inputs, outputs = output)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.MeanAbsolutePercentageError()
    metrics = ['accuracy']
    model.compile(optimizer, loss = loss)

    history = model.fit(X_train, X_train, epochs=200, batch_size=64, verbose=0)

    train_loss = history.history['loss'][-1]
    return train_loss

def latent_count(candidate):
    return candidate[-1]

def param_count():
    pass
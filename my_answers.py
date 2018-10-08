import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, CuDNNLSTM
import keras


# todo: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series)):
        if i+window_size < len(series):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# todo: build an RNN to perform regression on our time series input/output data

def build_part1_RNN(window_size):
    model = Sequential()
    model.add(CuDNNLSTM(5, input_shape=(window_size, 1)))
    # model.add(CuDNNLSTM(5))
    # model.add(LSTM(50, input_shape=(window_size, 1)))
    # model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))
    # model.add(LSTM(50))
    model.add(Dense(1))
    model.summary()

    return model


# todo: return the text input with only ascii lowercase and
# the punctuation given below included.

def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
    unique = sorted(list(set(text)))

    for i in unique:
        if i not in punctuation and i not in alphabet:
            text = text.replace(i, ' ')
    
    return text

# todo: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model

def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text), step_size):
        if i+window_size < len(text):
            inputs.append(text[i:i+window_size])
            outputs.append(text[i+window_size])

    return inputs, outputs

# todo build the required RNN model: 
# a single LSTM hidden layer with softmax activation, 
# categorical_crossentropy loss

def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    # model.add(CuDNNLSTM(5, input_shape=(window_size, 1)))
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars,  activation='linear'))
    model.add(Activation(activation='softmax'))

    return model

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def import_dataset(path):
    test1 = np.loadtxt(path)
    test1 = np.flip(test1, axis=0)
    dataset = preprocessing.MinMaxScaler(feature_range=(-1, 1)).\
        fit_transform(test1.reshape(-1, 1))
    dataset = dataset.reshape(-1)

    # lets take a look at our time series
    plt.plot(dataset)
    plt.xlabel('time period')
    plt.ylabel('normalized series value')
    plt.show()

    return dataset

def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series)):
        if i+window_size < len(series):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])

    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


def split_data(X, y, window_size, train_percent=0.8, valid=True):
    split_sets = {}
    if valid:
        train_split = int(np.ceil(len(y)*train_percent))
        valid_split = int(np.ceil(len(y)*((1-train_percent)/2))) + train_split

        # partition the training set
        split_sets['X_train'] = X[:train_split, :]
        split_sets['y_train'] = y[:train_split]

        split_sets['X_valid'] = X[train_split:valid_split, :]
        split_sets['y_valid'] = y[train_split:valid_split]

        # keep the last chunk for testing
        split_sets['X_test'] = X[valid_split:, :]
        split_sets['y_test'] = y[valid_split:]

        # NOTE: to use keras's RNN LSTM module our input must
        # be reshaped to [samples, window size, stepsize]
        split_sets['X_train'] = np.asarray(np.reshape(split_sets['X_train'],
            (split_sets['X_train'].shape[0], window_size, 1)))
        split_sets['X_valid'] = np.asarray(np.reshape(split_sets['X_valid'],
            (split_sets['X_valid'].shape[0], window_size, 1)))
        split_sets['X_test'] = np.asarray(np.reshape(split_sets['X_test'],
            (split_sets['X_test'].shape[0], window_size, 1)))

        return train_percent, split_sets
    else:
        train_split = int(np.ceil(len(y)*train_percent))

        # partition the training set
        split_sets['X_train'] = X[:train_split, :]
        split_sets['y_train'] = y[:train_split]

        # keep the last chunk for testing
        split_sets['X_test'] = X[train_split:, :]
        split_sets['y_test'] = y[train_split:]

        # NOTE: to use keras's RNN LSTM module our input must be
        # reshaped to [samples, window size, stepsize]
        split_sets['X_train'] = np.asarray(np.reshape(split_sets['X_train'],
            (split_sets['X_train'].shape[0], window_size, 1)))
        split_sets['X_test'] = np.asarray(np.reshape(split_sets['X_test'],
            (split_sets['X_test'].shape[0], window_size, 1)))

        return train_percent, split_sets

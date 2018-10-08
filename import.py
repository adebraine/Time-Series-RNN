import numpy as np

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

def split_data(dataset, window_size, train_percent=0.8, valid=True):
    if valid:
        train_split = int(np.ceil(len(y)*train_percent))
        valid_split = int(np.ceil(len(y)*(1-train_percent/2))) + train_split
        
        # partition the training set
        X_train = X[:train_split,:]
        y_train = y[:train_split]

        X_valid = X[train_split:valid_split,:]
        y_valid = y[train_split:valid_split]

        # keep the last chunk for testing
        X_test = X[valid_split:,:]
        y_test = y[valid_split:]
        
        # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
        X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
        X_valid = np.asarray(np.reshape(X_valid, (X_valid.shape[0], window_size, 1)))
        X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    else:
        train_split = int(np.ceil(len(y)*train_percent))
        
        # partition the training set
        X_train = X[:train_split,:]
        y_train = y[:train_split]

        # keep the last chunk for testing
        X_test = X[train_split:,:]
        y_test = y[train_split:]
        
        # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize] 
        X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
        X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))
        
        return X_train, y_train, X_test, y_test
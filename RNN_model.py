from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, CuDNNLSTM
import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output


def build_RNN(window_size):
    model = Sequential()
    model.add(CuDNNLSTM(10, input_shape=(window_size, 1)))
    # model.add(CuDNNLSTM(5))
    # model.add(LSTM(50, input_shape=(window_size, 1)))
    # model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))
    # model.add(LSTM(50))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(split_sets, model, epochs, callbacklist,
                batch_size, optimizer, verbose=0):
    predictions = {}
    if len(split_sets) > 4:
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        trained_model = model.fit(split_sets['X_train'], split_sets['y_train'],
                                  epochs=epochs,
                                  callbacks=callbacklist,
                                  validation_data=(split_sets['X_valid'],
                                                   split_sets['y_valid']),
                                  batch_size=batch_size, verbose=verbose)

        predictions['train'] = model.predict(split_sets['X_train'])
        predictions['test'] = model.predict(split_sets['X_test'])
        predictions['valid'] = model.predict(split_sets['X_valid'])
    else:
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        trained_model = model.fit(split_sets['X_train'], split_sets['y_train'],
                                  epochs=epochs,
                                  callbacks=callbacklist,
                                  batch_size=batch_size, verbose=verbose)

        predictions['train'] = model.predict(split_sets['X_train'])
        predictions['test'] = model.predict(split_sets['X_test'])

    return trained_model, predictions


# https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e
# Below is live plot
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.show()

class epochN(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_epochs = self.params['epochs']
        self.i = 0

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.i += 1
        clear_output(wait=True)
        print('Epoch {}/{}'.format(self.i, self.total_epochs))
        print('Training_loss = {}'.format(logs.get('loss')))

class epochN_val(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.total_epochs = self.params['epochs']
        self.i = 0
        self.val_losses = []
        self.max_val_loss = 0

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.max_val_loss = max(self.val_losses)
        if self.i%10 == 0:
            self.val_losses = []
        self.i += 1

        clear_output(wait=True)
        print('Epoch {}/{}'.format(self.i, self.total_epochs))
        print('Valid_loss = {}'.format(self.max_val_loss))
        print('Training_loss = {}'.format(logs.get('loss')))


# Below is plotting after the training is done
def plot_loss(model, valid=True):
    if valid:
        loss = model.history['loss']
        val_loss = model.history['val_loss']
        plt.plot(loss, label="loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.show()
    else:
        pass

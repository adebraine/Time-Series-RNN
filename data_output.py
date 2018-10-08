import numpy as np
import matplotlib.pyplot as plt
import keras


def evaluate_model(model, split_sets):
    training_error = model.evaluate(split_sets['X_train'], split_sets['y_train'], verbose=0)
    print('training error = ' + str(training_error))

    testing_error = model.evaluate(split_sets['X_test'], split_sets['y_test'], verbose=0)
    print('testing error = ' + str(testing_error))


def output_plot(dataset, y, window_size, train_percent,
                predictions):
    if len(predictions) > 2:
        train_split = int(np.ceil(len(y)*train_percent)) + window_size
        valid_split = int(np.ceil(len(y)*((1-train_percent)/2))) + train_split

        # plot original series
        plt.plot(dataset, color='k')

        # plot training set prediction
        plt.plot(np.arange(window_size, train_split, 1),
                 predictions['train'], color='b')

        # plot validation set prediction
        plt.plot(np.arange(train_split, valid_split, 1),
                 predictions['valid'], color='g')

        # plot testing set prediction
        plt.plot(np.arange(valid_split, valid_split + len(predictions['test']), 1),
                 predictions['test'], color='r')

        # pretty up graph
        plt.xlabel('day')
        plt.ylabel('(normalized) price')
        plt.legend(['original series', 'training fit',
                    'Validation fit', 'testing fit'],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    else:
        train_split = int(np.ceil(len(y)*train_percent)) + window_size

        # plot original series
        plt.plot(dataset, color='k')

        # plot training set prediction
        plt.plot(np.arange(window_size, train_split, 1),
                 predictions['train'], color='b')

        # plot testing set prediction
        plt.plot(np.arange(train_split, train_split + len(predictions['test']), 1),
                 predictions['test'], color='r')

        # pretty up graph
        plt.xlabel('day')
        plt.ylabel('(normalized) price')
        plt.legend(['original series', 'training fit',
                    'testing fit'],
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

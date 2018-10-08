from data_import import *
from data_output import *
from RNN_model import *


dataset = import_dataset('datasets/Bitcoin.csv')
window_size = 3
epochs = 1000
batch_size = 50
verbose = 0

# With Validation Set?
valid = 1

X, y = window_transform_series(series=dataset, window_size=window_size)
train_percent, split_sets = split_data(X, y, window_size, valid=valid)
# given - fix random seed - so we can all reproduce the same
# results on our default time series
np.random.seed(0)

callbacklist = []
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9,
                                     epsilon=1e-08, decay=0.0)

model = build_RNN(window_size)

trained_model = train_model(split_sets, model, epochs, callbacklist,
                            batch_size, optimizer, verbose=1)
predictions = predict_model(split_sets, model)
plot_loss(trained_model, valid)

# print out training and testing errors
evaluate_model(model, split_sets)

output_plot(dataset, y, window_size, train_percent, predictions)


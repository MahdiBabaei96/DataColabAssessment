# %cd '/content/drive/My Drive/SS'

import pandas as pd
import matplotlib.pyplot as plt
from numpy import array, hstack
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Bidirectional
from sklearn.metrics import precision_score, mean_absolute_error, mean_squared_error

# read the csv file
df = pd.read_csv('a.csv')

# multivariate output data prep 
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
  
# define input sequence
in_seq1 = df['west']
in_seq2 = df['central']
in_seq3 = df['east']
in_seq4 = df['north']
in_seq5 = df['south']
in_seq6 = df['national']

# convert to [rows, columns] structure
in_seq1 = in_seq1.values.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.values.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.values.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.values.reshape((len(in_seq4), 1))
in_seq5 = in_seq5.values.reshape((len(in_seq5), 1))
in_seq6 = in_seq6.values.reshape((len(in_seq6), 1))

# horizontally stack columns
dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6))

# choose a number of time steps
n_features = 6
n_steps = 5

# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)

# split the training and test set
x_train=X[:28000]
x_test=X[28000:]
y_train=y[:28000]
y_test=y[28000:]

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=False, input_shape=(n_steps, n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(x_train, y_train, epochs=300, batch_size=8, verbose=0)

# evaluation
y_pred = model.predict(x_test, verbose=0)
mae = mean_absolute_error(y_pred, y_test)
print('MAE: %f' % mae)
mse = mean_squared_error(y_pred, y_test)
print('MSE: %f' % mse)

# fine-tuning hyperparameters
"""from math import sqrt
from pandas import DataFrame
from matplotlib import pyplot
train_rmse, test_rmse = list(), list()
for i in range(300):
  model.fit(x_train, y_train, epochs=1, verbose=0)
  model.reset_states()
  yhat = model.predict(x_train, verbose=0)
  train_rmse.append(sqrt(mean_squared_error(yhat, y_train)))
  # evaluate model on test data
  model.reset_states()
  yhat = model.predict(x_test, verbose=0)
  test_rmse.append(sqrt(mean_squared_error(yhat, y_test)))
  history = DataFrame()
  history['train'], history['test'] = train_rmse, test_rmse
  pyplot.plot(history['train'], color='blue')
  pyplot.plot(history['test'], color='orange')
  print('%d) TrainRMSE=%f, TestRMSE=%f' % (i, history['train'].iloc[-1], history['test'].iloc[-1]))"""
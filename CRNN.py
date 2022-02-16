import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import csv
import sklearn
from google.colab import drive 

drive.mount('/content/gdrive')

# Predict one torque value at a time from 10000 light intensity values (5000 per sensor)
train_split = 0.9
val_split = 0.05

output_num = 1
input_shape = 10000
values_per_sensor = int((input_shape/2))
skip_values = 100

# Initialise input matrix X and output matrix Y
X = []
Y = []

def lightIntensityData(path):
  """Opens input light intensity data and returns it as two lists (sensor1 and sensor2)"""
  sensor1 = []
  sensor2 = []
  with open(path, newline = '\n') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    for row in data:
        sensor1.append(round(float(row[1]), 5))
        sensor2.append(round(float(row[2]), 5))
  return sensor1, sensor2

def realTorqueData(path):
  """Opens the torque data and returns it as a list (torque)"""
  torque = []
  with open(path, newline = '\n') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    torque.append(next(data)[0])
    for row in data:
      torque.append(round(float(row[0]), 5))
    torque1 = [part for part in torque[0]]
    torque[0] = round(float(''.join(torque1[1:])), 5)
  return torque

def appendData(x, y, sensor1, sensor2, torque):
  """Appends light intensity and torque data to the input and output data matrices
  (x holds input light intensity data and y holds the output real torque values)
  Intermediate r is used to alternate data from each light sensor in x"""
  for i in range(values_per_sensor, len(sensor1), skip_values):
    r1 = sensor1[i-(values_per_sensor-1):i+1]
    r2 = sensor2[i-(values_per_sensor-1):i+1]
    r = [None]*(len(r1)+len(r2))
    r[::2] = r1
    r[1::2] = r2
    x.append(r)
    y.append(torque[i])
  return x, y

def createDataSets(X_shuff, Y_shuff, trainSplit, valSplit):
  """Splits the shuffled data sets into training, validation and testing datasets
  as numpy arrays"""
  lenX_train = int(len(X_shuff)*trainSplit)
  lenY_train = int(len(Y_shuff)*trainSplit)

  lenX_val = int(len(X_shuff)*valSplit)
  lenY_val = int(len(Y_shuff)*valSplit)

  X_train = X_shuff[:lenX_train]
  Y_train = Y_shuff[:lenY_train]
  X_val = X_shuff[lenX_train:(lenX_train+lenX_val)]
  Y_val = Y_shuff[lenY_train:(lenY_train+lenY_val)]
  X_test = X_shuff[(lenX_train+lenX_val):]
  Y_test = Y_shuff[(lenY_train+lenY_val):]

  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  X_val = np.array(X_val)
  Y_val = np.array(Y_val)
  X_test = np.array(X_test)
  Y_test = np.array(Y_test)

  return X_train, Y_train, X_val, Y_val, X_test, Y_test

def makePrediction(modelNN, X, Y, offset, seq_len):
  """Uses modelNN to predict the torque starting at offset for seq_len values, and
  returning the predicted torque values (y_hat) along with the corresponding real values 
  (y_acc)"""
  y_hat = []
  y_acc = Y[offset:(offset+seq_len)]
  x = np.array(X)
  for i in range(seq_len):
    y = modelNN.predict(x[i+offset].reshape(1, input_shape, 1))
    y_hat.append(y[0])
  return y_hat, y_acc

def timeArray(skipValues, predictionLength, start):
  """Defines the x axis when plotting torque"""
  skipTime = 1*(10**-5)*skipValues
  begin = start*skipTime
  end = begin + predictionLength*skipTime
  return np.arange(begin, end, skipTime)

"""Custom callback to record loss of individual batches
put in callback_list when fitting to use"""
class BatchLoss(keras.callbacks.Callback):
  def on_train_begin(self):
    self.losses = []

  def on_batch_end(self, logs = {}):
    self.losses.append(logs.get('loss'))

batch_losses = BatchLoss()

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180410-0007.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/10-0007FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20171219-0004.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/19-0004FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180216-0006.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/16-0006FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180216-0008.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/16-0008FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180406-0006.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/06-0006FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180406-0008.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/06-0008FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180406-0009.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/06-0009FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

S1, S2 = lightIntensityData('/content/gdrive/MyDrive/Colab Notebooks/OP_20180406-00010.csv')
y = realTorqueData('/content/gdrive/MyDrive/Colab Notebooks/06-00010FILT1500.csv')
X, Y = appendData(X, Y, S1, S2, y)

# Shuffles data so that training, testing and validation data sets are representative
X_shuffled, Y_shuffled = sklearn.utils.shuffle(X, Y)

X_train, Y_train, X_val, Y_val, X_test, Y_test = createDataSets(X_shuffled, Y_shuffled, train_split, val_split)

# Defining the structure of the neural network
input_tensor = keras.Input(shape=(input_shape, 1))

x = keras.layers.Conv1D(96, 7, activation='relu')(input_tensor)
x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
x = keras.layers.Conv1D(256, 7, activation='relu')(x)
x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
x = keras.layers.Conv1D(384, 5, activation='relu')(x)
x = keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')(x)
x = keras.layers.Conv1D(384, 5, activation = 'relu')(x)
x = keras.layers.MaxPooling1D(pool_size=4, strides=4, padding='valid')(x)
x = keras.layers.GRU(1000, return_sequences = True)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.Dropout(rate=0.1)(x)
x = keras.layers.Dense(248, activation='relu')(x)

output_tensor = keras.layers.Dense(output_num, activation=None)(x)

# Defines the model
model = keras.models.Model(input_tensor, output_tensor)

# Compiles with MSE loss function as regression problem
# Adam optimiser used, metrics to be displayed during training 
model.compile(loss=tf.keras.losses.MeanSquaredError(),
 optimizer="Adam",
 metrics=['mse', keras.metrics.RootMeanSquaredError()])

# Early stopping and reducing learning rate on plateau callbacks
callback_list = [
  keras.callbacks.EarlyStopping(monitor='val_mse', patience=8, restore_best_weights=True),
  keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.1, patience=2, min_lr=(10^-7))
]

# print(model.summary())
history = model.fit(X_train, Y_train, batch_size=16, callbacks=callback_list, epochs=20, validation_data=(X_val, Y_val))

test_score = model.evaluate(X_test, Y_test, verbose=0)

model.save('/content/gdrive/MyDrive/Colab/CRNN_20_epoch.h5')

# model = keras.models.load_model('/content/gdrive/MyDrive/Colab/CRNN.h5')

start = 12000
length = 2000
Y_hat, Y_acc = makePrediction(model, X, Y, start, length);
xAxis = timeArray(skip_values, length, start)

plt.figure(figsize = (10,5), dpi=100)
plt.plot(xAxis, Y_acc, 'g')

# Y_hat predictions - red
# True values - green
plt.plot(xAxis, Y_hat, 'r')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()

# Training losses graph
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
# plt.gca().set_ylim(0, 1.2)
plt.xlabel("Epochs")
plt.show()
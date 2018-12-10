from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Dropout
import urllib
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
import time
from keras import callbacks
from keras.callbacks import TensorBoard
import tensorflow as tf

#Set seed
random.seed(10)
#Set start time
start_time = time.time()

#Importing Data
url_response = urllib.urlretrieve('https://storage.googleapis.com/ml2-group4-project/all_images.npy', 'all_images.npy')
x = np.load('all_images.npy')

url_response = urllib.urlretrieve('https://storage.googleapis.com/ml2-group4-project/all_labels.npy', 'all_labels.npy')
y = np.load('all_labels.npy')
y = y.astype(np.float32)


#Convolutional Network
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
#model.add(layers.MaxPooling2D(2,2))

#Fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
# model.add(Dropout(0.4))
model.add(layers.Dense(5, activation= 'softmax'))



#Split into train and test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

train_images = X_train
train_labels = y_train
test_images = X_test
test_labels = y_test

#Training convolutional network

from keras.utils import to_categorical

# print(train_images.shape)
# print(test_images.shape)

optimizer = optimizers.rmsprop(lr=0.001)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Plot tensorboard
tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
          write_graph=True, write_images=True)



model.fit(train_images, train_labels, epochs=5, batch_size=100, verbose=1, callbacks=[tbCallBack])


#Evaluate model on test data

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
print(test_loss)
print(time.time() - start_time)


#Visualization

#Plot model network
from keras.utils import plot_model
plot_model(model, to_file='model.png')

#Plot training history visualization

import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.3, epochs=5, batch_size=100)

# list all data in history
print(history.history.keys())

# # History of  accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # History of loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

# function to use of python 3 from use of python 2
from __future__ import print_function

# use keras library
import keras

# import module function 'mnist' from module name 'keras.datasets'
from keras.datasets import mnist

# sequence : a particular order in which related things follow each other
# import sequntial model from keras.models
from keras.models import Sequential

'''
dense : closely compacted in substance
dense layer : fully connected layer
dropout : loss
dropout layer : prevent overfitting to lose some layers
flatten : make flat
flatten in deep learning : the process of converting all the resultant 2 demensional arrays into a single long continuous linear vector
'''
from keras.layers import Dense, Dropout, Flatten

# Conv2D : constructs a two-dimensional "convolutional" layer
# MaxPooling2D : constructs a two-dimentional 'pooling' layer
from keras.layers import Conv2D, MaxPooling2D

# backend module : one of TensorFlow, Theano, CNTK
from keras import backend as K

# batch_size : number of test samples
batch_size = 128
# num_classes : not sure
num_classes = 10
# epochs : one pass of the full training set
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

#the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ?
if K.image_data_format() == 'channels_first':
    # ?
    # reshape : reshape an output to a certain shape
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test,shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    # ?    
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# astype : copy of the array, cast to a specified type
# float32 : about 7 decimal digits accuracy
# astype('float32') for division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# rgb color 255
# feature scaling
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# to_categorical : converts a class vector(integers) to binary class matrix
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_date=(x_test, y_test)
score = model.evaluate(x_test, y_test, verbose=0)
print('Tess loss:', score[0])
print('Test accuracy:', score[1])


'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

#function to use of python 3 from use of python 2
from __future__ import print_function

#use keras library
import keras

#import module function 'mnist' from module name 'keras.datasets'
from keras.datasets import mnist

#sequence : a particular order in which related things follow each other
#import sequntial model from keras.models
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

#Conv2D : constructs a two-dimensional "convolutional" layer
#MaxPooling2D : constructs a two-dimentional 'pooling' layer
from keras.layers import Conv2D, MaxPooling2D

#backend module : one of TensorFlow, Theano, CNTK
from keras import backend as K

#batch_size : number of test samples
batch_size = 128
#num_classes : not sure
num_classes = 10
#epochs : one pass of the full training set
epochs = 2


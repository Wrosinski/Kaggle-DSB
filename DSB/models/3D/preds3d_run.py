import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th') 

import numpy as np
import pandas as pd
import cv2
import zarr
import glob
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop, Nadam

from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout3D
from keras.utils.np_utils import to_categorical

from preds3d_utils import *
from preds3d_models import *


def preds3d_baseline(width):
    
    learning_rate = 5e-5
    optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    #optimizer = Adam(lr=learning_rate)
    
    inputs = Input(shape=(1, 136, 168, 168))
    conv1 = Convolution3D(width, 3, 3, 3, activation = 'relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv1)
    
    conv2 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv2)

    conv3 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), border_mode='same')(conv3)
    
    output = GlobalAveragePooling3D()(pool3)
    output = Dense(2, activation='softmax', name = 'predictions')(output)
    model3d = Model(inputs, output)
    model3d.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model3d


# 1398 stage1 original examples

width = 16
start_train = 0
end_train = 1398
start_val = 1400
end_val = 1595

epochs = 20


cnn3d_genfit('DSB_class3d_preds3dbaseline_sgd_1398pats', preds3d_baseline(width), epochs, start_train, end_train, start_val, end_val, 
             end_train - start_train,
             end_val - start_val)





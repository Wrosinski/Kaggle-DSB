import numpy as np 
import pandas as pd 
import skimage, os
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import glob
import zarr
from sklearn.utils import shuffle
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th') 

from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, merge, UpSampling2D
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout3D
from keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

from utils_3d import *


def unet_model():
    
    inputs = Input(shape=(1, max_slices, img_size, img_size))
    conv1 = Convolution3D(width, 3, 3, 3, activation = 'relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv1)
    conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), border_mode='same')(conv1)
    
    conv2 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv2)
    conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), border_mode='same')(conv2)

    conv3 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv3)
    conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2), strides = (2, 2, 2), border_mode='same')(conv3)
    
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*16, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)

    up5 = merge([UpSampling3D(size=(2, 2, 2))(conv4), conv3], mode='concat', concat_axis=1)
    conv5 = SpatialDropout3D(dropout_rate)(up5)
    conv5 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv5)
    conv5 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(conv5)
    
    up6 = merge([UpSampling3D(size=(2, 2, 2))(conv5), conv2], mode='concat', concat_axis=1)
    conv6 = SpatialDropout3D(dropout_rate)(up6)
    conv6 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv6)
    conv6 = Convolution3D(width*4, 3, 3, 3, activation = 'relu', border_mode='same')(conv6)

    up7 = merge([UpSampling3D(size=(2, 2, 2))(conv6), conv1], mode='concat', concat_axis=1)
    conv7 = SpatialDropout3D(dropout_rate)(up7)
    conv7 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv7)
    conv7 = Convolution3D(width*2, 3, 3, 3, activation = 'relu', border_mode='same')(conv7)
    conv8 = Convolution3D(1, 1, 1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv8)
    model.compile(optimizer=Adam(lr=1e-5), 
                  loss=dice_coef_loss, metrics=[dice_coef])

    return model


def generate_train(start, end, seed = None):
    size_3d = 136
    size = 168
    lungs = sorted(glob.glob(src + 'lung_mask/*.npy'))[start:end]
    nods = sorted(glob.glob(src + 'nodule_mask/*.npy'))[start:end]
    while True:
        print('Shuffling data')
        lungs, nods = shuffle(lungs, nods)
        for i in range(len(lungs)):
            lung_3d = np.full((1, 1, size_3d, size, size), 7.).astype('float32')
            nodule_3d = np.zeros((1, 1, size_3d, size, size)).astype('float32')
            lung = np.load(lungs[i]).astype('float32')
            nod = np.load(nods[i]).astype('float32')
            lung = lung.swapaxes(1, 0)
            nod = nod.swapaxes(1, 0)
            num_slices = lung.shape[1]
            offset = (size_3d - num_slices)
            if offset == 0:
                lung_3d[0, :, :, :, :] = lung[:, :, :, :]
                nodule_3d[0, :, :, :, :] = nod[:, :, :, :]
            if offset > 0:
                begin_offset = int(np.round(offset/2))
                end_offset = int(offset - begin_offset)
                lung_3d[0, :, begin_offset:-end_offset, :, :] = lung[:, :, :, :]
                nodule_3d[0, :, begin_offset:-end_offset, :, :] = nod[:, :, :, :]
            if offset < 0:
                print('{} slices lost due to size restrictions'.format(offset))
                offset = -(size_3d - num_slices)
                begin_offset = int(np.round(offset/2))
                end_offset = int(offset - begin_offset)
                lung_3d[0, :, :, :, :] = lung[:, begin_offset:-end_offset, :, :]
                nodule_3d[0, :, :, :, :] = nod[:, begin_offset:-end_offset, :, :]
                del lung, nod
                gc.collect()
            yield(lung_3d, nodule_3d)
            
def generate_val(start, end, seed = None):
    size_3d = 136
    size = 168
    lungs = sorted(glob.glob(src + 'lung_mask/*.npy'))[start:end]
    nods = sorted(glob.glob(src + 'nodule_mask/*.npy'))[start:end]
    while True:
        for i in range(len(lungs)):
            lung_3d = np.full((1, 1, size_3d, size, size), 7.).astype('float32')
            nodule_3d = np.zeros((1, 1, size_3d, size, size)).astype('float32')
            lung = np.load(lungs[i]).astype('float32')
            nod = np.load(nods[i]).astype('float32')
            lung = lung.swapaxes(1, 0)
            nod = nod.swapaxes(1, 0)
            num_slices = lung.shape[1]
            offset = (size_3d - num_slices)
            if offset == 0:
                lung_3d[0, :, :, :, :] = lung[:, :, :, :]
                nodule_3d[0, :, :, :, :] = nod[:, :, :, :]
            if offset > 0:
                begin_offset = int(np.round(offset/2))
                end_offset = int(offset - begin_offset)
                lung_3d[0, :, begin_offset:-end_offset, :, :] = lung[:, :, :, :]
                nodule_3d[0, :, begin_offset:-end_offset, :, :] = nod[:, :, :, :]
            if offset < 0:
                print('{} slices lost due to size restrictions'.format(offset))
                offset = -(size_3d - num_slices)
                begin_offset = int(np.round(offset/2))
                end_offset = int(offset - begin_offset)
                lung_3d[0, :, :, :, :] = lung[:, begin_offset:-end_offset, :, :]
                nodule_3d[0, :, :, :, :] = nod[:, begin_offset:-end_offset, :, :]
                del lung, nod
                gc.collect()
            yield(lung_3d, nodule_3d)


def unet_fit(name, start_t, end_t, start_v, end_v, check_name = None):
    
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 15, 
                                   verbose = 1),
    ModelCheckpoint('/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    
    if check_name is not None:
        check_model = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(check_name)
        model = load_model(check_model, 
                           custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model()

    model.fit_generator(generate_train(start_t, end_t), nb_epoch = 150, verbose = 1, 
                        validation_data = generate_val(start_v, end_v), 
                        callbacks = callbacks,
                        samples_per_epoch = 551, nb_val_samples = 50)
        
    return


# In[5]:

src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/'
max_slices = 136
img_size = 168
dropout_rate = 0.5
width = 8

img_rows = img_size
img_cols = img_size


#unet_fit('3DUNet_genfulldata_patients_merged', 0, 551, 551, 601)
unet_fit('3DUNet_genfulldata_patients_merged_cont', 0, 551, 551, 601, '3DUNet_genfulldata_patients_merged')

# In[ ]:




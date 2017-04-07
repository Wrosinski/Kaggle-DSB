import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th') 

import numpy as np
import pandas as pd
import cv2
import zarr
import glob
import time
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop, Nadam

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout3D
from keras.utils.np_utils import to_categorical


# In[2]:

zarr_dir = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr/'
zarr_store = zarr.DirectoryStore(zarr_dir)
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')
dsb_pats = os.listdir('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr/nodules_candidates/')


def load_zarr(patient_id):
    lung_cand_zarr = zarr_load_group['nodules_candidates'][patient_id]
    return np.array(lung_cand_zarr).astype('float32')

def load_data(start, end):
    print('Loading 2D full-size candidates.')
    df = pd.read_csv('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_labels_full2.csv')[start:end]
    df = df[df['id'].isin(dsb_pats)]
    t = time.time()
    masks = np.zeros((0, 1, 512, 512))
    labels = np.zeros(0)
    for i in range(len(df)):
        mask = load_zarr('{}'.format(df.iloc[i, 0]))
        mask[mask <= 1.0] = 0.
        z_nonzero = np.unique(np.nonzero(mask)[0])
        mask = mask[z_nonzero[0]:z_nonzero[-1], :, :]
        print('Nonzero mask shape:', mask.shape[0])
        mask = mask/255.
        masks = np.concatenate((masks, mask), 0)
        if df.iloc[i, 1] == 1:
            label = np.ones(mask.shape[0])
        if df.iloc[i, 1] == 0:
            label = np.zeros(mask.shape[0])
        labels = np.concatenate((labels, label), 0)
    print('Data shape:', masks.shape)
    print('Time it took to load the data:', time.time() - t)
    return masks, df, labels


def check_shapes(start, end):
    print('Loading 2D full-size candidates.')
    df = pd.read_csv('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_labels_full2.csv')[start:end]
    df = df[df['id'].isin(dsb_pats)]
    full_shape = 0
    for i in range(len(df)):
        mask = load_zarr('{}'.format(df.iloc[i, 0]))
        full_shape += mask.shape[0]
    return full_shape


def generate_train(start, end, batch_size, shuffle = True):
    step = 10
    subsets = np.arange(start, end, step)
    while True:
        for s in range(len(subsets)-1):
            print(subsets[s], subsets[s+1])
            masks, _, labels = load_data(subsets[s], subsets[s+1])
            y = to_categorical(labels, 2)
            #y = y.reshape((-1,1))
            sample_index = np.arange(0, masks.shape[0])
            
            if shuffle:
                print('Shuffling data')
                perm = np.random.permutation(len(sample_index))
                masks = masks[perm]
                y = y[perm]
            if len(sample_index) % batch_size != 0:
                mod = len(sample_index) % batch_size
                sample_index = sample_index[:-mod]
                assert len(sample_index) % batch_size == 0
            for j in range(len(sample_index)//batch_size):
                batch_index = sample_index[j*batch_size:j*batch_size+batch_size]
                lung_batch = masks[batch_index, :, :, :]
                y_batch = y[batch_index]
                yield(lung_batch, y_batch)

                
def generate_val(start, end, batch_size):
    step = 10
    subsets = np.arange(start, end, step)
    while True:
        for s in range(len(subsets)-1):
            print(subsets[s], subsets[s+1])
            masks, _, labels = load_data(subsets[s], subsets[s+1])
            y = to_categorical(labels, 2)
            #y = y.reshape((-1,1))
            sample_index = np.arange(0, masks.shape[0])
            
            if len(sample_index) % batch_size != 0:
                mod = len(sample_index) % batch_size
                sample_index = sample_index[:-mod]
                assert len(sample_index) % batch_size == 0
            for j in range(len(sample_index)//batch_size):
                batch_index = sample_index[j*batch_size:j*batch_size+batch_size]
                lung_batch = masks[batch_index, :, :, :]
                y_batch = y[batch_index]
                yield(lung_batch, y_batch)



def cnn2d():
    width = 4
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(inputs)
    #conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(conv1)
    #conv1 = BatchNormalization(axis = 1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(pool1)
    #conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(conv2)
    #conv2 = BatchNormalization(axis = 1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(pool2)
    #conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(conv3)
    #conv3 = BatchNormalization(axis = 1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width*16, 3, 3, activation='relu', border_mode='same')(pool3)
    #conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Convolution2D(width*16, 3, 3, activation='relu', border_mode='same')(conv4)
    #conv4 = BatchNormalization(axis = 1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), border_mode='same')(conv4)
    
    output = Flatten(name='flatten')(pool4)
    output = Dropout(0.2)(output)
    output = Dense(256)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(128)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(2, activation='softmax', name = 'predictions')(output)

    model = Model(input=inputs, output=output)
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr = 1e-4), metrics = ['accuracy'])
    return model


def cnn_genfit(name, batch_size, cnn, samples_tr, samples_val, 
               start_tr, end_tr, start_val, end_val):
    
    callbacks = [EarlyStopping(monitor='val_loss', patience = 3, 
                                   verbose = 1),
    ModelCheckpoint('/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
   
    model = cnn()
    model.fit_generator(generate_train(start_tr, end_tr, batch_size), 
                        nb_epoch = 25, verbose = 1, callbacks = callbacks, 
                        samples_per_epoch = samples_tr,
                        validation_data = generate_val(start_val, end_val, batch_size),
                        nb_val_samples = samples_val)
    return


#samplestr = check_shapes(0, 1398)
#samplesval = check_shapes(1398, 1594)

samplestr = 443092
samplesval = 61992

cnn_genfit('1stgentry_CNN2DClassifier', 8, cnn2d, samplestr, samplesval, 0, 1402, 1398, 1598)





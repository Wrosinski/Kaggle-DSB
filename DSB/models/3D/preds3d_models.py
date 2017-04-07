from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.optimizers import Adam, SGD, RMSprop, Nadam
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers.advanced_activations import PReLU
from keras.layers import BatchNormalization, GlobalAveragePooling3D, GlobalMaxPooling3D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import SpatialDropout3D


def preds3d_baseline(width):
    
    learning_rate = 5e-5
    #optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    optimizer = Adam(lr=learning_rate)
    
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

 
def preds3d_globalavg(width):
    
    learning_rate = 5e-5
    #optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    optimizer = Adam(lr=learning_rate)
    
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
    
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*16, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    pool4 = MaxPooling3D(pool_size=(8, 8, 8), border_mode='same')(conv4)
    
    output = GlobalAveragePooling3D()(conv4)
    output = Dense(2, activation='softmax', name = 'predictions')(output)
    model3d = Model(inputs, output)
    model3d.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model3d


def preds3d_dense(width):
    
    learning_rate = 5e-5
    #optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    optimizer = Adam(lr=learning_rate)
    
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
    
    conv4 = Convolution3D(width*8, 3, 3, 3, activation = 'relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution3D(width*16, 3, 3, 3, activation = 'relu', border_mode='same')(conv4)
    conv4 = BatchNormalization(axis = 1)(conv4)
    pool4 = MaxPooling3D(pool_size=(8, 8, 8), border_mode='same')(conv4)
    
    output = Flatten(name='flatten')(pool4)
    output = Dropout(0.2)(output)
    output = Dense(128)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(128)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(2, activation='softmax', name = 'predictions')(output)
    model3d = Model(inputs, output)
    model3d.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model3d

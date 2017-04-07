from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D
from keras.layers import Input, merge, UpSampling2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

smooth = 1.0
dropout_rate = 0.35
width = 32


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def unet_model1():
    inputs = Input((1, 512, 512))
    conv1 = Convolution2D(width, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = BatchNormalization(axis = 1)(conv1)
    conv1 = Convolution2D(width, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = BatchNormalization(axis = 1)(conv2)
    conv2 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = BatchNormalization(axis = 1)(conv3)
    conv3 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = BatchNormalization(axis = 1)(conv4)
    conv4 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(width*16, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = BatchNormalization(axis = 1)(conv5)
    conv5 = Convolution2D(width*16, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = SpatialDropout2D(dropout_rate)(up6)
    conv6 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(conv6)
    conv6 = Convolution2D(width*8, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = SpatialDropout2D(dropout_rate)(up7)
    conv7 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(conv7)
    conv7 = Convolution2D(width*4, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = SpatialDropout2D(dropout_rate)(up8)
    conv8 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(conv8)
    conv8 = Convolution2D(width*2, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = SpatialDropout2D(dropout_rate)(up9)
    conv9 = Convolution2D(width, 3, 3, activation='relu', border_mode='same')(conv9)
    conv9 = Convolution2D(width, 3, 3, activation='relu', border_mode='same')(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model

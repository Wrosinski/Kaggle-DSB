import numpy as np 
import pandas as pd 
from scipy import ndimage
import time
import scipy
import gc
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th') 

from unet_utils import *
from unet_models import *
from inception_unet import *
np.random.seed(1337)



def unet_fit(name, batch_size, unet_model, load_check = False, check_name = None):
    
    t = time.time()
    callbacks = [EarlyStopping(monitor='val_loss', patience = 5, 
                                   verbose = 1),
    ModelCheckpoint('/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    if load_check:
        check_model = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(check_name)
        model = load_model(check_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    else:
        model = unet_model()
        
    model.fit_generator(generate_train(lung_path, nodule_path, start_t, end_t, batch_size), nb_epoch = 25, verbose = 1, validation_data = generate_val(lung_path, nodule_path, batch_size), callbacks = callbacks, samples_per_epoch = f1, nb_val_samples = f2)
    return

lung_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_subsets/lung_mask/'
nodule_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_subsets/nodule_mask/'

start_t, end_t = 0, 11
start_v, end_v = end_t, end_t + 1
f1 = get_max_slices(lung_path, start_t, end_t)
f2 = get_max_slices(lung_path, start_v, end_v)
f1 = f1
f2 = f2//2
print('Training samples:', f1, 'Validation samples:', f2)

smooth = 1.0
img_rows = 512
img_cols = 512
batch_size = 6

unet_fit('Unet1_fulldatagen', batch_size, unet_model1)

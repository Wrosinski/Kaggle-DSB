import numpy as np
import pandas as pd
import cv2
import zarr
import glob
import matplotlib.pyplot as plt
import os
import glob
import time
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model


zarr_dir = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr/'
zarr_store = zarr.DirectoryStore(zarr_dir)
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')
dsb_pats = os.listdir('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr/cands_resized/')


def print_cands(cand):
    plt.figure(figsize=(8,8))
    plt.imshow(cand, cmap = 'bone')
    return  

def generate_train(start, end, seed = None):
    df = pd.read_csv('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_labels_full.csv')[start:end]
    while True:
        print('Shuffling df')
        df = df.sample(frac=1).reset_index(drop=True)
        labels = to_categorical(df['cancer'])
        for i in range(len(df)):
            cand = load_zarr('{}'.format(df.iloc[i, 0]))
            y = labels[[i]]
            yield(cand, y)
            
def generate_val(start, end, seed = None):
    df = pd.read_csv('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_labels_full.csv')[start:end]
    while True:
        labels = to_categorical(df['cancer'])
        for i in range(len(df)):
            cand = load_zarr('{}'.format(df.iloc[i, 0]))
            y = labels[[i]]
            yield(cand, y)
            

def load_zarr(patient_id):
    lung_cand_zarr = zarr_load_group['cands_resized'][patient_id]
    return np.array(lung_cand_zarr).astype('float32')

def load_data(start, end):
    print('Loading 2D U-Net candidates.')
    df = pd.read_csv('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_labels.csv')[start:end]
    t = time.time()
    cands = np.zeros((0, 1, 136, 168, 168), dtype = 'float32')
    for i in range(len(df)):
        cand = load_zarr('{}'.format(df['id'][i]))
        cands = np.concatenate((cands, cand), 0)
    print('Data shape:', cands.shape)
    print('Time it took to load the data:', time.time() - t)
    return cands, df


def cnn3d_genfit(name, nn_model, epochs, start_t, end_t, start_v, end_v, nb_train, nb_val, check_name = None):
    callbacks = [EarlyStopping(monitor='val_loss', patience = 15, 
                                   verbose = 1),
    ModelCheckpoint('/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(name), 
                        monitor='val_loss', 
                        verbose = 0, save_best_only = True)]
    if check_name is not None:
        check_model = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(check_name)
        model = load_model(check_model)
    else:
        model = nn_model
    model.fit_generator(generate_train(start_t, end_t), nb_epoch = epochs, verbose = 1, 
                        validation_data = generate_val(start_v, end_v), 
                        callbacks = callbacks,
                        samples_per_epoch = nb_train, nb_val_samples = nb_val)
    return

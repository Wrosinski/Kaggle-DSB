
# coding: utf-8

# In[1]:

import numpy as np
import os
import glob
import pandas as pd
import time
from joblib import Parallel, delayed
from keras.preprocessing.image import ImageDataGenerator
import zarr
from unet_utils import *

def prep_subset(subset, savename, zarr = False):
    lung_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_2d/lung_mask/'
    nodule_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_2d/nodule_mask/'
    train = sorted([x for x in os.listdir(lung_path) if '.npy' in x])
    masks = np.zeros((0, 1, 512, 512))
    nodules = np.zeros((0, 1, 512, 512))
    t = time.time()
    for i in subset:
        mask = np.load(lung_path + train[i]).astype('float32')
        nod = np.load(nodule_path + train[i]).astype('float32')
        masks = np.concatenate((masks, mask), 0)
        nodules = np.concatenate((nodules, nod), 0)
    print('Data shape:', masks.shape)
    print('Time it took to load the data:', time.time() - t)
    masks = my_PreProc(masks)
    nodules[nodules == 1.0] = 255.
    print('Data preprocessed')
    if zarr:
        save_zarr(savename, lung_mask_group_proc, nodule_mask_group_proc, masks, nodules)
    else:
        np.save('{}/lung_mask/{}'.format(dst, savename), masks)
        np.save('{}/nodule_mask/{}'.format(dst, savename), nodules)
    return


def augment_subset(start = 0, end = 12, num_aug = 100, zarr = True):
    lung_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_subsets/lung_mask/'
    nodule_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_subsets/nodule_mask/'
    train = sorted([x for x in os.listdir(lung_path)])[start:end]
    for i in range(len(train)):
        t = time.time()
        masks = np.load(lung_path + train[i]).astype('float32')
        nodules = np.load(nodule_path + train[i]).astype('float32')
        t2 = time.time()
        print('Loading data took:', t2 - t)
        print('Original data shape:', masks.shape)
        masks_aug, nodules_aug = augmentation(masks, nodules, num_aug)
        assert masks_aug.shape == nodules_aug.shape
        print('Augmented data shape:', masks_aug.shape)
        print('Augmenting data took:', time.time() - t2, '\n')
        if zarr:
            s = 'subset_aug_{}'.format(i)
            save_zarr(s, lung_mask_group_aug, nodule_mask_group_aug, masks_aug, nodules_aug)
        else:
            np.save('{}/lung_mask/subset_aug_{}'.format(dst_aug, i), masks_aug)
            np.save('{}/nodule_mask/subset_aug_{}'.format(dst_aug, i), nodules_aug)
        
    return


def run_saving():
    subsets = []
    step = 100
    start = 0
    end = step
    for i in range(12):
        subsets.append(np.arange(start, end, 1))
        start += step
        end += step
    subsets[-1] = subsets[-1][:-14]
    for i, val in enumerate(subsets):
        prep_subset(val, 'subset_{}'.format(i))
    return


dst = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_subsets/'

run_saving()
#augment_subset(num_aug = 1500)

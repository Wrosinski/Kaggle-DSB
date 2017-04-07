import numpy as np 
import pandas as pd 
import skimage, os
from scipy import ndimage
import time
import scipy
import gc
import os
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
K.set_image_dim_ordering('th') 
np.random.seed(1337)

smooth = 1.0



def load_data(start, end):
    mask_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_2d/lung_mask/'
    nodules_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_2d/nodule_mask/'
    patients = sorted(os.listdir(mask_path))[start:end]
    t = time.time()
    masks = np.zeros((0, 1, 512, 512))
    nodules = np.zeros((0, 1, 512, 512))
    for i in patients:
        mask = np.load(mask_path + i)
        nod = np.load(nodules_path + i)
        masks = np.concatenate((masks, mask), 0)
        nodules = np.concatenate((nodules, nod), 0)
    print('Data shape:', masks.shape)
    print('Time it took to load the data:', time.time() - t)
    return masks, nodules

def get_max_slices(lung_path, start, end):
    subsets = os.listdir(lung_path)[start:end]
    full_slices = 0
    for i in range(len(subsets)):
        num_slices = np.load(lung_path + subsets[i]).shape[0]
        full_slices += num_slices
    print('Number of 2D slices in CT image: {}'.format(full_slices))
    return full_slices


def generate_train(lung_path, nodule_path, start, end, batch_size, shuffle = True):
    train = sorted([x for x in os.listdir(lung_path)])
    train = train[start:end]
    while True:
        for i in range(len(train)):
            lung_mask = np.load(lung_path + train[i])
            nodule_mask = np.load(nodule_path + train[i])
            assert lung_mask.shape == nodule_mask.shape
            sample_index = np.arange(0, lung_mask.shape[0])
            if shuffle:
                print('Shuffling data')
                perm = np.random.permutation(len(sample_index))
                lung_mask = lung_mask[perm]
                nodule_mask = nodule_mask[perm]
            if len(sample_index) % batch_size != 0:
                mod = len(sample_index) % batch_size
                lung_aug, nodule_aug = augmentation(lung_mask, nodule_mask, mod+1)
                sample_index = np.arange(0, lung_aug.shape[0])
            else:
                lung_aug, nodule_aug = lung_mask, nodule_mask
            del lung_mask, nodule_mask
            gc.collect()
            for j in range(len(sample_index)//batch_size):
                batch_index = sample_index[j*batch_size:j*batch_size+batch_size]
                lung_batch = lung_aug[batch_index, :, :, :]
                nodule_batch = nodule_aug[batch_index, :, :, :]
                yield(lung_batch, nodule_batch)

def generate_val(lung_path, nodule_path, batch_size):
    val = sorted([x for x in os.listdir(lung_path)])
    val = val[-1:]
    while True:
        lung_mask = np.load(lung_path + val[0])
        nodule_mask = np.load(nodule_path + val[0])
        assert lung_mask.shape == nodule_mask.shape
        sample_index = np.arange(0, lung_mask.shape[0])
        if len(sample_index) % batch_size != 0:
            mod = len(sample_index) % batch_size
            lung_aug, nodule_aug = augmentation(lung_mask, nodule_mask, mod+1)
            sample_index = np.arange(0, lung_aug.shape[0])
        else:
            lung_aug, nodule_aug = lung_mask, nodule_mask
        del lung_mask, nodule_mask
        gc.collect()
        for j in range(len(sample_index)//batch_size):
            batch_index = sample_index[j*batch_size:j*batch_size+batch_size]
            lung_batch = lung_aug[batch_index, :, :, :]
            nodule_batch = nodule_aug[batch_index, :, :, :]
            yield(lung_batch, nodule_batch)



def augmentation(scans,masks,n):
    datagen = ImageDataGenerator(
        featurewise_center=False,   
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=25,   
        width_shift_range=0.3,  
        height_shift_range=0.3,   
        horizontal_flip=True,   
        vertical_flip=True,  
        zoom_range=False)
    i=0
    for batch in datagen.flow(scans, batch_size=1, seed=1000): 
        scans=np.vstack([scans,batch])
        i += 1
        if i > n:
            break
    i=0
    for batch in datagen.flow(masks, batch_size=1, seed=1000): 
        masks=np.vstack([masks,batch])
        i += 1
        if i > n:
            break
    return((scans,masks))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i][0], cmap=plt.cm.bone)
    return
        
def print_mask(lung_m, nodule_m):
    fig, ax = plt.subplots(1,2, figsize=(20,16))
    ax[0].imshow(lung_m, cmap = plt.cm.bone)
    ax[1].imshow(nodule_m, cmap = plt.cm.bone)
    return



def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    #imgs_std = np.std(imgs)
    imgs_mean = 0.25 #np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)
    #imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
        imgs_normalized[i][imgs_normalized[i] > 255.] = 255.
        imgs_normalized[i][imgs_normalized[i] < 0.] = 0.
    return imgs_normalized

def my_PreProc(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==1)  #Use the original images
    train_imgs = dataset_normalized(data)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.25)
    #train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs

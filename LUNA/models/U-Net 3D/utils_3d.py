import numpy as np 
import pandas as pd 
import skimage, os
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import zarr
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')
from PIL import Image
import cv2

def weight_by_class_balance(truth, classes=None):
    if classes is None:
        classes = np.unique(truth)
    weight_map = np.zeros_like(truth, dtype=np.float32)
    total_amount = np.product(truth.shape)
    for c in classes:
        class_mask = np.where(truth==c,1,0)
        class_weight = 1/((np.sum(class_mask)+1e-8)/total_amount)
        weight_map += (class_mask*class_weight)#/total_amount
    return weight_map

def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
    return
        
def print_mask3d(lung_m, nodule_m):
    fig, ax = plt.subplots(1,2, figsize=(10,8))
    ax[0].imshow(lung_m, cmap = plt.cm.bone)
    ax[1].imshow(nodule_m, cmap = plt.cm.bone)
    return
    
def get_max_slices(start, end):
    mask_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/lung_mask/'
    nodules_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/nodule_mask/'
    patients = os.listdir(mask_path)[start:end]
    max_slices = 0
    full_slices = 0
    for i in range(len(patients)):
        num_slices = np.load(nodules_path + patients[i]).astype('float16').shape[0]
        full_slices += num_slices
        if num_slices > max_slices:
            max_slices = num_slices
    print('Number of max slices in CT image: {}'.format(max_slices))
    print('Number of 2D slices in CT image: {}'.format(full_slices))
    return max_slices, full_slices


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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
    scans_g=scans.copy()
    for batch in datagen.flow(scans, batch_size=1, seed=1000): 
        scans_g=np.vstack([scans_g,batch])
        i += 1
        if i > n:
            break
    i=0
    masks_g=masks.copy()
    for batch in datagen.flow(masks, batch_size=1, seed=1000): 
        masks_g=np.vstack([masks_g,batch])
        i += 1
        if i > n:
            break
    return((scans_g,masks_g))

def predict_segments(model,scans):
    pred = model.predict(scans_g, verbose=1)
    for i in range(scans.shape[0]):
        print ('scan '+str(i))
        f, ax = plt.subplots(1, 2,figsize=(10,5))
        ax[0].imshow(scans[i,0,:,:],cmap=plt.cm.gray)
        ax[1].imshow(pred[i,0,:,:],cmap=plt.cm.gray)
        plt.show()
 
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

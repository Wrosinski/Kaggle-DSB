import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import cv2
from unet_utils import *
import zarr
import glob
import time
import matplotlib.pyplot as plt
from keras.models import load_model


def load_zarr(patient_id):
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    return np.array(lung_mask_zarr)

def save_cands(id_patient, cands):
    candidates.array(id_patient, cands, 
            chunks=(40, 1, 512, 512), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())


def load_unet(check_name):
    check_model = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/Scripts/LUNA/CNN/Checkpoints/{}.h5'.format(check_name)
    model = load_model(check_model, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    return model


zarr_store = zarr.DirectoryStore('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr')
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
predictions = zarr_group.require_group('nodules_predictions')
candidates = zarr_group.require_group('nodules_candidates')

zarr_dir = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr'
zarr_store = zarr.DirectoryStore(zarr_dir)
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')



def predict_dsbpats(start, end):
    model = load_unet('2DUnet_genfulldata')
    ids = sorted([x for x in os.listdir(zarr_dir + '/lung_mask/') if 'z' not in x])[start:end]
    for i in range(len(ids)):
        print('Predicting for patient :', ids[i])
        t = time.time()
        lung = load_zarr(ids[i])
        preds = model.predict(lung, batch_size = 8)
        print('Time it took to predict: ', time.time() - t)
        cands = lung * preds
        cands = cands.astype('float32')
        try:
            save_cands(ids[i], cands)
        except KeyError:
            print('Patient {} already saved.'.format(ids[i]))
                  
        print('Time it took to predict & save: ', time.time() - t)
    return 


# 100-107 cands first done.
# + 0-5 done
# + 107-500 done
predict_dsbpats(500, 1595)

import os
import numpy as np
import cv2
from unet_utils import *
import zarr
import glob
import time
from scipy import ndimage
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def load_zarr(patient_id):
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    lung_cand_zarr = zarr_load_group['nodules_candidates'][patient_id]
    return np.array(lung_mask_zarr), np.array(lung_cand_zarr)

def save_cands(id_patient, cands):
    cands_resized.array(id_patient, cands, 
            chunks=(1, 17, 21, 21), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2),
            synchronizer=zarr.ThreadSynchronizer())


def get_masks(dsb_pats):
    thres = 1.
    size_3d = 136
    height = width = 168
    for i in dsb_pats:
        masks = np.zeros((1, 1, size_3d, height, width), dtype = 'float32')
        t1 = time.time()
        mask = load_zarr(i)[1]
        print('Candidates original shape:', mask.shape)
        print('Loaded candidates for patient:', i)
        mask[mask <= thres] = 0.
        mask = mask/255.
        mask = mask.swapaxes(1, 0)
        mask = ndimage.interpolation.zoom(mask, (1, 0.32, 0.32, 0.32))
        print('Candidates resized shape:', mask.shape)
        mask[mask <= 0.] = 0.
        z_offset = (size_3d - mask.shape[1])
        xy_offset = (height - mask.shape[2])
        print('Z-axis Offset:', z_offset, 'X & Y-axis Offset:', xy_offset)
        if z_offset == 0:
            begin_offset_xy = int(np.round(xy_offset/2))
            end_offset_xy = int(xy_offset - begin_offset_xy)
            masks[0, :, :, begin_offset_xy:-end_offset_xy, begin_offset_xy:-end_offset_xy] = mask[:, :, :]
        if z_offset > 0:
            begin_offset = int(np.round(z_offset/2))
            end_offset = int(z_offset - begin_offset)
            begin_offset_xy = int(np.round(xy_offset/2))
            end_offset_xy = int(xy_offset - begin_offset_xy)
            masks[0, :, begin_offset:-end_offset, begin_offset_xy:-end_offset_xy, begin_offset_xy:-end_offset_xy] = mask[:, :, :, :]
        if z_offset < 0:
            offset = -(size_3d - mask.shape[1])
            begin_offset = int(np.round(z_offset/2))
            end_offset = int(z_offset - begin_offset)
            begin_offset_xy = int(np.round(xy_offset/2))
            end_offset_xy = int(xy_offset - begin_offset_xy)
            masks[0, :, :, begin_offset_xy:-end_offset_xy, begin_offset_xy:-end_offset_xy] = mask[:, begin_offset:-end_offset, :, :]
        print('Time it took to resize and prepare patient {} : {}'.format(i, time.time() - t1))
        save_cands(i, masks.astype('float32'))
        print('Time it took to resize, prepare and save patient {} : {}'.format(i, time.time() - t1))
    return


zarr_store = zarr.DirectoryStore('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr')
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
cands_resized = zarr_group.require_group('cands_resized')
    
zarr_dir = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr'
zarr_store = zarr.DirectoryStore(zarr_dir)
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')

dsb_pats = sorted([x for x in os.listdir('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr/nodules_candidates/') if '.' not in x])

# First 301 done
# First 1288 done

get_masks(dsb_pats[1288:1289])
#Parallel(n_jobs=5)(delayed(get_masks)(patient) for patient in dsb_pats[301:])



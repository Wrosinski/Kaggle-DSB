import numpy as np
import zarr
import glob
import os
import time

def save_zarr(id_patient, lung_mask, nodule_mask):
    lung_mask_group.array(id_patient, lung_mask, 
            chunks=(10, 1, 512, 512), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    nodule_mask_group.array(id_patient, nodule_mask, 
            chunks=(10, 1, 512, 512), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    return

def load_zarr(patient_id):
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    nodule_mask_zarr = zarr_load_group['nodule_mask'][patient_id]
    return lung_mask_zarr, nodule_mask_zarr 


zarr_store = zarr.DirectoryStore('/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/processed_zarr/2d.zarr')
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
lung_mask_group = zarr_group.require_group('lung_mask')
nodule_mask_group = zarr_group.require_group('nodule_mask')

src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_2d/'
lungs_npy = glob.glob(src + 'lung_mask/*.npy')
nodules_npy = glob.glob(src + 'nodule_mask/*.npy')

ids = os.listdir(src + 'lung_mask/')[10:]

def load_and_save():
    for i in range(len(ids)):
        print('Saving patient: ', ids[i])
        t = time.time()
        lung = np.load(lungs_npy[i])
        nod = np.load(nodules_npy[i])
        save_zarr(ids[i], lung, nod)
        print('Time it took to load & save: ', time.time() - t)
    return

load_and_save()

import numpy as np
import zarr
import glob
import os
import time

def save_zarr(id_patient, lung_mask, cand):
    lung_mask_group.array(id_patient, lung_mask, 
            chunks=(1, 17, 21, 21), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    cand_group.array(id_patient, cand, 
            chunks=(1, 17, 21, 21), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    return

def load_zarr(patient_id):
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    nodule_mask_zarr = zarr_load_group['candidates'][patient_id]
    return lung_mask_zarr, nodule_mask_zarr 


zarr_store = zarr.DirectoryStore('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_3d_zarr/3d.zarr')
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
lung_mask_group = zarr_group.require_group('lung_mask')
cand_group = zarr_group.require_group('candidates')

src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_3d_npy/'
lungs_npy = glob.glob(src + 'lung_mask/*.npy')
cands_npy = glob.glob(src + 'candidates/*.npy')

ids = os.listdir(src + 'lung_mask/')

def load_and_save():
    for i in range(len(ids)):
        print('Saving patient: ', ids[i])
        t = time.time()
        lung = np.load(lungs_npy[i])
        cand = np.load(cands_npy[i])
        save_zarr(ids[i], lung, cand)
        print('Time it took to load & save: ', time.time() - t)
    return

load_and_save()

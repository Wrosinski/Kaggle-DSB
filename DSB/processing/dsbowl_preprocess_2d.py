import SimpleITK as sitk
import numpy as np
import csv
import scipy
from glob import glob
import pandas as pd
from scipy import ndimage
from tqdm import tqdm 


import pandas as pd
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np
from skimage import measure, morphology, segmentation

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from numba import autojit
import zarr

from PIL import Image
import cv2
from dsbowl_utils import *

# In[ ]:

def load_train():
    patients = os.listdir(src)
    patients.sort()
    return patients
        
def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone) 
         


# In[ ]:

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    pos1 = slices[int(len(slices)/2)].ImagePositionPatient[2]
    pos2 = slices[(int(len(slices)/2)) + 1].ImagePositionPatient[2]
    diff = pos2 - pos1
    if diff > 0:
        slices = np.flipud(slices)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices
    
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image <= threshold_min] = 0
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)
    return np.array(image, dtype=np.int16)
    
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    if spacing[0] == 0.0:
        print('Z-axis spacing missing, substitute with Y-axis spacing.')
        spacing[0] = spacing[1]
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    print('Original spacing : ', spacing, 'Resizing factor: ', real_resize_factor)
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


# In[ ]:

def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros(image.shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    return marker_internal, marker_external, marker_watershed


def get_segmented_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    
    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)
    
    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)
    
    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)
    
    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    #blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 14) # <- retains more of the area, 12 works well. Changed to 14, 12 still excluded some parts.
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)
    
    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)
    
    #Apply the lungfilter (note the filtered areas being assigned threshold_min HU)
    segmented = np.where(lungfilter == 1, image, threshold_min*np.ones(image.shape))
    
    #return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed
    return segmented

def reshape_3d(image_3d):
    reshaped_img = image_3d.reshape([image_3d.shape[0], 1, 512, 512])
    print('Reshaped image shape:', reshaped_img.shape)
    return reshaped_img




def create_masks_for_patient_watershed(patient_id, save = False):

    print("Getting mask for patient {}".format(patient_id))
    dicom_file = src + patient_id
    patient = load_scan(dicom_file)
    lung_orig = get_pixels_hu(patient)
    lung_img, spacing = resample(lung_orig, patient, [1,1,1])
    print('Original image shape: {}'.format(lung_orig.shape))
    print('Resized image shape: {}'.format(lung_img.shape))

    lung_mask = lung_img.copy()
    #lung_mask[lung_mask == 0] = threshold_min
    lung_mask[lung_mask >= threshold_max] = threshold_max
    lung_img[lung_img >= threshold_max] = threshold_max

    lung_mask_512 = np.zeros((lung_mask.shape[0], 512, 512))
    lung_mask_512[lung_mask_512 == 0] = threshold_min

    original_shape = lung_img.shape
    for z in range(lung_img.shape[0]):
        offset = (512 - original_shape[1])
        upper_offset = int(np.round(offset/2))
        lower_offset = int(offset - upper_offset)
        lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]

    lung_mask_512 = reshape_3d(lung_mask_512)
    lung_mask_512[lung_mask_512 <= threshold_min] = threshold_min
    
    lung_mask_512 = my_PreProc(lung_mask_512)
    lung_mask_512 = lung_mask_512.astype(np.float32)

    if save:
        try:
            save_zarr(patient_id, lung_mask_512)
        except KeyError:
            print('Failed for patient: ', patient_id)

    return


def save_zarr(id_patient, lung_mask):
    lung_mask_group.array(id_patient, lung_mask, 
            chunks=(10, 1, 512, 512), compressor=zarr.Blosc(clevel=9, cname="zstd", shuffle=2), 
            synchronizer=zarr.ThreadSynchronizer())
    return

def load_zarr(patient_id):
    lung_raw_zarr = zarr_load_group['lung_raw'][patient_id]
    lung_mask_zarr = zarr_load_group['lung_mask'][patient_id]
    nodule_mask_zarr = zarr_load_group['nodule_mask'][patient_id]
    return lung_raw_zarr, lung_mask_zarr, nodule_mask_zarr

def scan_show(img):
    plt.figure(figsize = (16,16))
    plt.imshow(img[int(img.shape[0]/2)][0], cmap = 'bone')
    return



zarr_store = zarr.DirectoryStore('/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_zarr/DSBowl.zarr')
zarr_group = zarr.hierarchy.open_group(store=zarr_store, mode='a')
lung_mask_group = zarr_group.require_group('lung_mask')
zarr_load_group = zarr.hierarchy.open_group(store=zarr_store, mode='r')


src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1/'
dst_processed = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/input_data/stage1_processed/'

patients = load_train()

threshold_min = -2000
threshold_max = 400

# All done!
def run():
    for i in patients[1137:1138]:
        create_masks_for_patient_watershed(i, True)
    return
run()

#Parallel(n_jobs=5)(delayed(create_masks_for_patient_watershed)(patient, True) for patient in patients[1135:1143])

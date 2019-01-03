#!/usr/bin/python

"""
Adapted from the original C++ example: densecrf/examples/dense_inference.cpp
http://www.philkr.net/home/densecrf Version 2.2
"""

import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
import sys
import glob
import os
from joblib import Parallel, delayed
import multiprocessing

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_single_denseCRF(image, saliency_map, add_half_salmap=False):

    EPSILON = 1e-8

    img = image
    annos = saliency_map


    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1,:]

    #res *= 255 / res.max()
    res = res * 255
    res = res.reshape(img.shape[:2])

    if add_half_salmap:
        res = res / 2 + annos / 2
    return res


def perform_dense_CRF(im_list_dict, idx, saliency_ext, saliency_folder, base_folder, output_folder):
    elem = im_list_dict[idx]
    image = cv2.imread(elem['full_path'], 1)
    saliency_im_name = elem['name'] + '_' + saliency_folder
    saliency_im_name_full = os.path.join(base_folder, saliency_folder, saliency_im_name + saliency_ext)
    saliency_map = cv2.imread(saliency_im_name_full, 0)
    result = compute_single_denseCRF(image, saliency_map, add_half_salmap=True)
    output_im_name_full = os.path.join(base_folder, output_folder, saliency_im_name + '_CRF' + saliency_ext)
    cv2.imwrite(output_im_name_full, result.astype('uint8'))
    print('im # {} done'.format(idx))

    return



def main():

    # Initialize the folders and file parameters
    # Name of the parent folder of the folder containing the images and saliency
    base_folder = 'C:/Users/dobeac/Documents/Git Projects/Saliency/Datasets/ECSSD'

    # name of the folder containing the original images
    image_folder = 'images'

    # name of the folder containing the saliency results
    saliency_folder = 'my_DSS_GDM5'

    # Name of the folder to create with the saliency results
    output_folder = saliency_folder + '_CRF'

    # File extensions
    im_ext = '.jpg'
    saliency_ext = '.png'
    output_folder_full = os.path.join(base_folder, output_folder)
    if not os.path.exists(output_folder_full):
        os.makedirs(output_folder_full)

    # Get the full files list
    im_list = glob.glob(os.path.join(base_folder, image_folder) + '/*' + im_ext)
    im_list_dict = []
    for idx, elem in enumerate(im_list):
        path, full_name = os.path.split(elem)
        name, _ = full_name.split('.')
        im_list_dict.append({'full_path': elem, 'path': path, 'full_name': full_name, 'name': name})

    # Loop all the images to compute the denseCRF and output a result
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(perform_dense_CRF)(im_list_dict, idx, saliency_ext, saliency_folder, base_folder, output_folder) for idx in range(0, len(im_list_dict)))



if __name__ == '__main__':
    main()
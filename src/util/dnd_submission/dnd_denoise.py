 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.

import numpy as np
import scipy.io as sio
import os
import h5py
from PIL import Image
from os.path import join
import numpy as np

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_raw(info, img_id, bb, yy, xx):
    nlf_h5 = info[info["sigma_raw"][0][img_id]]
    sigma = nlf_h5[xx,yy,bb]
    return sigma

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma

def denoise_raw(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all raw images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the parameters of the noise level
                  function (nlf["a"], nlf["b"]) and a mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_raw', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['Inoisy']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3]].copy()
            Idenoised_crop = Inoisy_crop.copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)
            for yy in range(2):
                for xx in range(2):
                    nlf["sigma"] = load_sigma_raw(info, i, k, yy, xx)
                    Inoisy_crop_c = Inoisy_crop[yy:H:2,xx:W:2].copy()
                    Idenoised_crop_c = denoiser(Inoisy_crop_c, nlf)
                    Idenoised_crop[yy:H:2,xx:W:2] = Idenoised_crop_c
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))

def denoise_srgb(denoiser, data_folder, out_folder):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            nlf = load_nlf(info, i)

            nlf["sigma"] = load_sigma_srgb(info, i, k)
            Idenoised_crop = denoiser(Inoisy_crop, nlf, i, k)
            
            # save denoised data
            Idenoised_crop = np.float32(Idenoised_crop)
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))


if __name__ == '__main__':
    denoise_srgb(denoiser=None, data_folder='AP-BSN/dataset/DND/dnd_2017', out_folder=None)
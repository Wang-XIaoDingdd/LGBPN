import os
from os.path import join
from glob import glob
import h5py
import scipy.io as sio
from PIL import Image
import numpy as np
import mat73
from tqdm import tqdm


srgb_mat = '../../dataset/SIDD/BenchmarkNoisyBlocksSrgb.mat'
raw_mat = '../../dataset/SIDD/BenchmarkNoisyBlocksRaw.mat'
benchmark_path = '../../dataset/SIDD/SIDD_Benchmark_Data'

id = lambda x: int(x.split('/')[-1].split('_')[0])


def pack_srgb_to_mat(path):
    filepath = srgb_mat
    img = sio.loadmat(filepath)
    Inoisy = np.float32(np.array(img['BenchmarkNoisyBlocksSrgb']))
    Inoisy /=255.
    restored = np.zeros_like(Inoisy)
    imgs = glob(join(path, '*DN*.png'))
    print('img len:', len(imgs))
    imgs = sorted(imgs)
    print('len', len(imgs))
    for i in tqdm(range(40)):
        for k in range(32):
            restored[i,k,:,:,:] = np.array(Image.open(imgs[i*32 + k]))
    restored = restored.astype(np.uint8)
    sio.savemat(os.path.join(path, 'Idenoised.mat'), {"Idenoised": restored,})


def from_npy_to_mat(from_path, to_path):
    img = np.load(from_path, allow_pickle=True)
    sio.savemat(to_path, {"x": img,})


if __name__ == '__main__':

    pack_srgb_to_mat('your_denoised_result_path')
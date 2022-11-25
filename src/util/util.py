from math import exp

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import scipy.io as sio
import mat73
import h5py


def np2tensor(n: np.array):
    '''
    transform numpy array (image) to torch Tensor
    BGR -> RGB
    (h,w,c) -> (c,h,w)
    '''
    # gray
    if len(n.shape) == 2:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(n, (2, 0, 1))))
    # RGB -> BGR
    elif len(n.shape) == 3:
        return torch.from_numpy(np.ascontiguousarray(np.transpose(np.flip(n, axis=2), (2, 0, 1))))
    else:
        raise RuntimeError('wrong numpy dimensions : %s' % (n.shape,))


def tensor2np(t: torch.Tensor, flip=True):
    '''
    transform torch Tensor to numpy having opencv image form.
    RGB -> BGR
    (c,h,w) -> (h,w,c)
    '''
    t = t.cpu().detach()

    # gray
    if flip:
        if len(t.shape) == 2:
            return t.permute(1, 2, 0).numpy()
        # RGB -> BGR
        elif len(t.shape) == 3:
            return np.flip(t.permute(1, 2, 0).numpy(), axis=2)
        # image batch
        elif len(t.shape) == 4:
            return np.flip(t.permute(0, 2, 3, 1).numpy(), axis=3)
        else:
            raise RuntimeError('wrong tensor dimensions : %s' % (t.shape,))

    else:
        if len(t.shape) == 2:
            return t.permute(1, 2, 0).numpy()
        # RGB -> BGR
        elif len(t.shape) == 3:
            return t.permute(1, 2, 0).numpy()
        # image batch
        elif len(t.shape) == 4:
            return t.permute(0, 2, 3, 1).numpy()
        else:
            raise RuntimeError('wrong tensor dimensions : %s' % (t.shape,))

def imwrite_tensor(t, name='test.png'):
    cv2.imwrite('./%s' % name, tensor2np(t.cpu()))


def imread_tensor(name='test'):
    return np2tensor(cv2.imread('./%s' % name))


def rot_hflip_img(img: torch.Tensor, rot_times: int = 0, hflip: int = 0):
    '''
    rotate '90 x times degree' & horizontal flip image 
    (shape of img: b,c,h,w or c,h,w)
    '''
    b = 0 if len(img.shape) == 3 else 1
    # no flip
    if hflip % 2 == 0:
        # 0 degrees
        if rot_times % 4 == 0:
            return img
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b + 1).transpose(b + 1, b + 2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b + 2).flip(b + 1)
        # 270 degrees
        else:
            return img.flip(b + 2).transpose(b + 1, b + 2)
    # horizontal flip
    else:
        # 0 degrees
        if rot_times % 4 == 0:
            return img.flip(b + 2)
        # 90 degrees
        elif rot_times % 4 == 1:
            return img.flip(b + 1).flip(b + 2).transpose(b + 1, b + 2)
        # 180 degrees
        elif rot_times % 4 == 2:
            return img.flip(b + 1)
        # 270 degrees
        else:
            return img.transpose(b + 1, b + 2)


def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c,w + 2 * f * pad,h + 2 * f * pad)
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c, w + 2 * f * pad, h + 2 * f * pad)


def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
        # batched image tensor
    else:
        b, c, w, h = x.shape
        before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f,
                                                                                              h // f)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def pixel_shuffle_down_sampling_pd(x: torch.Tensor, f: int, pad: int = 0, pad_value: float = 0.):
    '''
    pixel-shuffle down-sampling (PD) from "When AWGN-denoiser meets real-world noise." (AAAI 2019)
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad between each down-sampled images
        pad_value (float) : padding value
    Return:
        pd_x (Tensor) : down-shuffled image tensor with pad or not
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # unshuffled = F.pixel_unshuffle(x, f)
        # if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), 'reflect')
        unshuffled = unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 2, 3, 1, 4, 5).contiguous()
        unshuffled = unshuffled.view(-1, c, w // f + 2 * pad, h // f + 2 * pad).contiguous()
        return unshuffled
        


def pixel_shuffle_up_sampling_pd(x: torch.Tensor, f: int, pad: int = 0):
    '''
    inverse of pixel-shuffle down-sampling (PD)
    see more details about PD in pixel_shuffle_down_sampling()
    Args:
        x (Tensor) : input tensor
        f (int) : factor of PD
        pad (int) : number of pad will be removed
    '''
    # single image tensor
    if len(x.shape) == 3:
        # c, w, h = x.shape
        # before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        # if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        # return -1
        pass
    # batched image tensor
    else:
        b, c, w, h = x.shape
        b = b // (f * f)
        before_shuffle = x.view(b, f, f, c, w, h)
        before_shuffle = before_shuffle.permute(0, 3, 1, 2, 4, 5).contiguous()
        before_shuffle = before_shuffle.view(b, c*f*f, w, h)
        if pad != 0: before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def psnr(img1, img2, data_range=255):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping & chnage type to uint8
    img1 = np.clip(img1, 0, data_range)
    img2 = np.clip(img2, 0, data_range)

    return peak_signal_noise_ratio(img1, img2, data_range=data_range)


def ssim(img1, img2, data_range=255):
    '''
    image value range : [0 - 255]
    clipping for model output
    '''
    if len(img1.shape) == 4:
        img1 = img1[0]
    if len(img2.shape) == 4:
        img2 = img2[0]

    # tensor to numpy
    if isinstance(img1, torch.Tensor):
        img1 = tensor2np(img1)
    if isinstance(img2, torch.Tensor):
        img2 = tensor2np(img2)

    # numpy value cliping
    img2 = np.clip(img2, 0, data_range)
    img1 = np.clip(img1, 0, data_range)

    return structural_similarity(img1, img2, multichannel=True, data_range=data_range)


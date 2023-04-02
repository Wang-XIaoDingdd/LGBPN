
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, pixel_shuffle_up_sampling_pd, pixel_shuffle_down_sampling_pd
# from src.util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, pixel_shuffle_up_sampling_pd, pixel_shuffle_down_sampling_pd

from . import regist_model
from .DBSNl import LGBPN
from PIL import Image
import numpy as np
from os.path import join
from einops import rearrange, repeat


@regist_model
class BSN(nn.Module):
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, SIDD=True,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p
        self.SIDD = SIDD

        # define network
        if bsn == 'My_BSN':
            self.bsn = LGBPN(in_ch, in_ch, bsn_base_ch, bsn_num_module, group=1, head_ch=24, SIDD=SIDD)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)

    def forward(self, img, pd=None, dict=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        pd_img_denoised = self.bsn(img, dict=dict)

        return pd_img_denoised

    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b, c, h, w = x.shape
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)

        # Random Replacing Refinement
        if self.R3 is False:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:, :, :h, :w]
        elif self.R3 is True:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input, refine=True)
                else:
                    denoised[..., t] = self.bsn(tmp_input, refine=True)[:, :, p:-p, p:-p]

            return torch.mean(denoised, dim=-1)


@regist_model
class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p

        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))

        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)

        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        return img_pd_bsn

    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b, c, h, w = x.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:, :, :h, :w]
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]

            return torch.mean(denoised, dim=-1)


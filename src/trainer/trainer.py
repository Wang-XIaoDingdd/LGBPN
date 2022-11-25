import os
import datetime
import random

import torch

from . import regist_trainer
from .base import BaseTrainer
from ..model import get_model_class
from functools import partial
from PIL import Image
import numpy as np
from einops import rearrange

@regist_trainer
class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @torch.no_grad()
    def test(self):
        ''' initialization test setting '''        
        # initialization
        dataset_load = (self.cfg['test_img'] is None) and (self.cfg['test_dir'] is None)
        self._before_test(dataset_load=dataset_load)

        # set image save path
        for i in range(60):
            test_time = datetime.datetime.now().strftime('%m-%d-%H-%M') + '-%02d'%i
            img_save_path = 'img/test_%s_%03d_%s' % (self.cfg['test']['dataset'], self.epoch, test_time)
            if not self.file_manager.is_dir_exist(img_save_path): break

        # -- [ TEST Single Image ] -- #
        if self.cfg['test_img'] is not None:
            self.test_img(self.cfg['test_img'])
            exit()
        # -- [ TEST Image Directory ] -- #
        elif self.cfg['test_dir'] is not None:
            self.test_dir(self.cfg['test_dir'])
            exit()
        # -- [ TEST DND Benchmark ] -- #
        elif self.test_cfg['dataset'] == 'DND_benchmark':
            self.test_DND(img_save_path)
            exit()
        # -- [ Test Normal Dataset ] -- #
        else:
            psnr, ssim = self.test_dataloader_process(  dataloader    = self.test_dataloader['dataset'], 
                                                        add_con       = 0.  if not 'add_con' in self.test_cfg else self.test_cfg['add_con'], 
                                                        floor         = False if not 'floor' in self.test_cfg else self.test_cfg['floor'],
                                                        img_save_path = img_save_path, 
                                                        img_save      = self.test_cfg['save_image'])
            # print out result as filename
            if psnr is not None and ssim is not None:
                with open(os.path.join(self.file_manager.get_dir(img_save_path), '_psnr-%.3f_ssim-%.4f.result'%(psnr, ssim)), 'w') as f:
                    f.write('PSNR: %f\nSSIM: %f'%(psnr, ssim))

    @torch.no_grad()
    def validation(self):
        # set denoiser
        self._set_denoiser()

        # make directories for image saving
        img_save_path = 'img/val_%03d' % self.epoch
        self.file_manager.make_dir(img_save_path)

        # validation
        psnr, ssim = self.test_dataloader_process(  dataloader    = self.val_dataloader['dataset'],
                                                    add_con       = 0.  if not 'add_con' in self.val_cfg else self.val_cfg['add_con'],
                                                    floor         = False if not 'floor' in self.val_cfg else self.val_cfg['floor'],                                                    
                                                    img_save_path = img_save_path,
                                                    img_save      = self.val_cfg['save_image'])

    def _set_module(self):
        module = {}
        if self.cfg['model']['kwargs'] is None:
            module['denoiser'] = get_model_class(self.cfg['model']['type'])()
        else:   
            module['denoiser'] = get_model_class(self.cfg['model']['type'])(**self.cfg['model']['kwargs'])
        return module
        
    def _set_optimizer(self):
        optimizer = {}
        for key in self.module:
            optimizer[key] = self._set_one_optimizer(opt        = self.train_cfg['optimizer'], 
                                                     parameters = self.module[key].parameters(), 
                                                     lr         = float(self.train_cfg['init_lr']))
        return optimizer

    def _forward_fn(self, module, loss, data):
        # forward
        input_data = [data['dataset'][arg] for arg in self.cfg['model_input']]
        denoised_img = module['denoiser'](*input_data)
        model_output = {'recon': denoised_img}


        # get losses
        losses, tmp_info = loss(input_data, model_output, data['dataset'], module, \
                                    ratio=(self.epoch-1 + (self.iter-1)/self.max_iter)/self.max_epoch)

        return losses, tmp_info



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

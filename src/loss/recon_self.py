import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import regist_loss
from PIL import Image
eps = 1e-6


# ============================ #
#  Self-reconstruction loss    #
# ============================ #

@regist_loss
class self_L1():
    def __call__(self, input_data, model_output, data, module):
        if isinstance(model_output, dict):
            output = model_output['recon']
        else:
            output = model_output
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.l1_loss(output, target_noisy)


@regist_loss
class self_L2():
    def __call__(self, input_data, model_output, data, module):
        output = model_output['recon']
        target_noisy = data['syn_noisy'] if 'syn_noisy' in data else data['real_noisy']

        return F.mse_loss(output, target_noisy)

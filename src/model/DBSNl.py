import torch
import torch.nn as nn
import torch.nn.functional as F

from . import regist_model
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, pixel_shuffle_up_sampling_pd, pixel_shuffle_down_sampling_pd, DeformConv2d
from PIL import Image
import numpy as np
import math
import torch.nn.parallel as P
from src.model.restormer_arch import DTB



class local_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module, group=1, pattern=0, head_ch=None, SIDD=True):
        super().__init__()

        kernel = 4 * stride + 1  #kernel=9
        pad = kernel // 2

        if head_ch is None:
            head_ch = in_ch

        self.Maskconv = DSPMC_9(head_ch, in_ch, kernel_size=kernel, stride=1, padding=pad, groups=group, padding_mode='reflect')

        ly = []
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)
        self.SIDD = SIDD

    def forward(self, x, refine=False, dict=None):
        
        if self.SIDD:
            pd_train_br1 = 4
            pd_test_br1 = 3
            pd_refine_br1 = 2
        else:
            pd_train_br1 = 4
            pd_test_br1 = 2
            pd_refine_br1 = 2

        if dict is not None:
            pd_test_br1 = dict['pd_test_br1']
            pd_refine_br1 = dict['pd_refine_br1']

        b, c, h, w = x.shape

        pad = 0
        if self.training:
            if h % pd_train_br1 != 0:
                x = F.pad(x, (0, 0, 0, pd_train_br1 - h % pd_train_br1), mode='reflect')
                pad = pd_train_br1 - h % pd_train_br1
            if w % pd_train_br1 != 0:
                x = F.pad(x, (0, pd_train_br1 - w % pd_train_br1, 0, 0), mode='reflect')
        elif not refine:
            if h % pd_test_br1 != 0:
                x = F.pad(x, (0, 0, 0, pd_test_br1 - h % pd_test_br1), mode='reflect')
                pad = pd_test_br1 - h % pd_test_br1
            if w % pd_test_br1 != 0:
                x = F.pad(x, (0, pd_test_br1 - w % pd_test_br1, 0, 0), mode='reflect')
        else:
            if h % pd_refine_br1 != 0:
                x = F.pad(x, (0, 0, 0, pd_refine_br1 - h % pd_refine_br1), mode='reflect')
                pad = pd_refine_br1 - h % pd_refine_br1
            if w % pd_refine_br1 != 0:
                x = F.pad(x, (0, pd_refine_br1 - w % pd_refine_br1, 0, 0), mode='reflect')

        x = self.Maskconv(x, refine, dict=dict, SIDD=self.SIDD)

        if self.training:
            x = pixel_shuffle_down_sampling(x, f=pd_train_br1, pad=2)
            x = self.body(x)
            x = pixel_shuffle_up_sampling(x, f=pd_train_br1, pad=2)
        # 原来的
        elif not refine:
            x = pixel_shuffle_down_sampling_pd(x, f=pd_test_br1, pad=2)
            x = self.body(x)
            x = pixel_shuffle_up_sampling_pd(x, f=pd_test_br1, pad=2)
        else:
            if pd_refine_br1 > 1:
                x = pixel_shuffle_down_sampling_pd(x, f=pd_refine_br1, pad=2)
                x = self.body(x)
                x = pixel_shuffle_up_sampling_pd(x, f=pd_refine_br1, pad=2)
            else:
                x = self.body(x)

        if pad != 0:
            x = x[:, :, :-pad, :-pad]

        return x


class global_branch(nn.Module):
    def __init__(self, stride, in_ch, num_module, group=1, head_ch=None, SIDD=True):
        super().__init__()

        kernel = 21
        pad = kernel // 2

        if head_ch is None:
            head_ch = in_ch

        self.Maskconv = DSPMC_21(head_ch, in_ch, kernel_size=kernel, stride=1, padding=pad, groups=group, padding_mode='reflect')

        self.body = DTB(stride=stride, num_blocks=num_module, dim=in_ch)
        self.SIDD = SIDD

    def forward(self, x, refine=False, dict=None):
        
        if self.SIDD:
            pd_train_br2 = 5
            pd_test_br2 = 4
            pd_refine_br2 = 4
        else:
            pd_train_br2 = 5
            pd_test_br2 = 4
            pd_refine_br2 = 2


        if dict is not None:
            pd_test_br2 = dict['pd_test_br2']
            pd_refine_br2 = dict['pd_refine_br2']

        b, c, h, w = x.shape

        pad = 0
        if self.training:
            if h % pd_train_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_train_br2 - h % pd_train_br2), mode='reflect')
                pad = pd_train_br2 - h % pd_train_br2
            if w % pd_train_br2 != 0:
                x = F.pad(x, (0, pd_train_br2 - w % pd_train_br2, 0, 0), mode='reflect')
        elif not refine:
            if h % pd_test_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_test_br2 - h % pd_test_br2), mode='reflect')
                pad = pd_test_br2 - h % pd_test_br2
            if w % pd_test_br2 != 0:
                x = F.pad(x, (0, pd_test_br2 - w % pd_test_br2, 0, 0), mode='reflect')
        else:
            if h % pd_refine_br2 != 0:
                x = F.pad(x, (0, 0, 0, pd_refine_br2 - h % pd_refine_br2), mode='reflect')
                pad = pd_refine_br2 - h % pd_refine_br2
            if w % pd_refine_br2 != 0:
                x = F.pad(x, (0, pd_refine_br2 - w % pd_refine_br2, 0, 0), mode='reflect')

        x = self.Maskconv(x, refine, dict=dict, SIDD=self.SIDD)
        if self.training:
            x = pixel_shuffle_down_sampling(x, f=pd_train_br2, pad=0)
            x = self.body(x)
            x = pixel_shuffle_up_sampling(x, f=pd_train_br2, pad=0)
        elif not refine:
            x = pixel_shuffle_down_sampling_pd(x, f=pd_test_br2, pad=7)
            x = self.body(x)
            x = pixel_shuffle_up_sampling_pd(x, f=pd_test_br2, pad=7)
        else:
            if pd_refine_br2 > 1:
                x = pixel_shuffle_down_sampling_pd(x, f=pd_refine_br2, pad=7)
                x = self.body(x)
                x = pixel_shuffle_up_sampling_pd(x, f=pd_refine_br2, pad=7)
            else:
                x = self.body(x)

        if pad != 0:
            x = x[:, :, :-pad, :-pad]

        return x



class DSPMC_9(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO:
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        # self.mask.fill_(0)

        kwargs_test = kwargs
        kwargs_test['stride'] = kW
        kwargs_test['padding'] = (0, 0)
        self.test_conv = nn.Conv2d(*args, **kwargs_test)
        # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)

        self.mask.fill_(1)

        dis = kH // 2
        for i in range(kH):
            for j in range(kW):
                if abs(i-dis) + abs(j-dis) <= dis:
                    self.mask[:, :, i, j] = 0

        a = 1
        # self.mask.detach().cpu().numpy()[0,0,...]


    def forward(self, x, refine=False, dict=None, SIDD=True):

        if SIDD:
            pd_test_ratio = 0.72
            pd_refine_ratio = 0.46
        else:
            pd_test_ratio = 0.5
            pd_refine_ratio = 0.4

        if dict is not None:
            pd_test_ratio = dict['pd_test_ratio']
            pd_refine_ratio = dict['pd_refine_ratio']

        if self.training:
            self.weight.data *= self.mask
            return super().forward(x)

        elif not refine:
            x_out = self.forward_chop(x, ratio=pd_test_ratio)
            return x_out

        else:
            x_out = self.forward_chop(x, ratio=pd_refine_ratio)
            return x_out


    def forward_chop(self, *args, shave=8, min_size=200000, n_GPUs=1, ratio=1):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1
        n_GPUs = torch.cuda.device_count()
        n_GPUs = min(n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h // 2 + shave)
        bottom = slice(h - h // 2 - shave, h)
        left = slice(0, w // 2 + shave)
        right = slice(w - w // 2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                weight = self.weight
                bias = self.bias
                inc, outc, kH, kW = self.weight.size()

                deform_conv = DeformConv2d(inc=inc, outc=outc, kernel_size=kW, stride=1, padding=kW//2, ratio=ratio)

                x_offset = P.data_parallel(deform_conv, *x, range(n_GPUs))
                # x_offset = deform_conv(x[0])

                # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)
                self.test_conv.weight = weight
                self.test_conv.bias = bias
                y = P.data_parallel(self.test_conv, x_offset, range(n_GPUs))

                # y = out
                del x_offset

                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size, ratio=ratio)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        bottom_r = slice(h // 2 - h, None)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)
        right_r = slice(w // 2 - w, None)

        if w % 2 != 0:
            right_r = slice(w // 2 - w + 1, None)
            bottom_r = slice(h // 2 - h + 1, None)
        else:
            right_r = slice(w // 2 - w, None)
            bottom_r = slice(h // 2 - h, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y


class DSPMC_21(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO:
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        # self.mask.fill_(0)

        kwargs_test = kwargs
        kwargs_test['stride'] = kW
        kwargs_test['padding'] = (0, 0)
        self.test_conv = nn.Conv2d(*args, **kwargs_test)
        # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)

        self.mask.fill_(0)

        stride = 2
        self.mask[:, :, ::stride, ::stride] = 1

        dis = 9 // 2

        for i in range(kH):
            for j in range(kW):
                if abs(i-kH//2) + abs(j-kW//2) <= dis:
                    self.mask[:, :, i, j] = 0

        a = 1
        # self.mask.detach().cpu().numpy()[0,0,...]


    def forward(self, x, refine=False, dict=None, SIDD=True):

        if SIDD:
            pd_test_ratio = 0.8
            pd_refine_ratio = 0.43
        else:
            pd_test_ratio = 0.65
            pd_refine_ratio = 0.35


        if dict is not None:
            pd_test_ratio = dict['pd_test_ratio_br2']
            pd_refine_ratio = dict['pd_refine_ratio_br2']

        if self.training:
            self.weight.data *= self.mask
            return super().forward(x)

        elif not refine:
            x_out = self.forward_chop(x, ratio=pd_test_ratio)
            return x_out

        else:
            x_out = self.forward_chop(x, ratio=pd_refine_ratio)
            return x_out

    # 之前是30
    def forward_chop(self, *args, shave=11, min_size=80000, n_GPUs=1, ratio=1):
        # scale = 1 if self.input_large else self.scale[self.idx_scale]
        scale = 1
        n_GPUs = torch.cuda.device_count()
        n_GPUs = min(n_GPUs, 4)
        # height, width
        h, w = args[0].size()[-2:]

        top = slice(0, h // 2 + shave)
        bottom = slice(h - h // 2 - shave, h)
        left = slice(0, w // 2 + shave)
        right = slice(w - w // 2 - shave, w)
        x_chops = [torch.cat([
            a[..., top, left],
            a[..., top, right],
            a[..., bottom, left],
            a[..., bottom, right]
        ]) for a in args]

        y_chops = []
        if h * w < 4 * min_size:
            for i in range(0, 4, n_GPUs):
                x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
                weight = self.weight
                bias = self.bias
                inc, outc, kH, kW = self.weight.size()

                deform_conv = DeformConv2d(inc=inc, outc=outc, kernel_size=kW, stride=1, padding=kW//2, ratio=ratio)

                x_offset = P.data_parallel(deform_conv, *x, range(n_GPUs))
                # x_offset = deform_conv(x[0])

                # y = F.conv2d(input=x_offset, weight=weight, bias=bias, stride=kW, groups=1)
                self.test_conv.weight = weight
                self.test_conv.bias = bias
                y = P.data_parallel(self.test_conv, x_offset, range(n_GPUs))

                # y = out
                del x_offset

                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y):
                        y_chop.extend(_y.chunk(n_GPUs, dim=0))
        else:
            for p in zip(*x_chops):
                y = self.forward_chop(*p, shave=shave, min_size=min_size, ratio=ratio)
                if not isinstance(y, list): y = [y]
                if not y_chops:
                    y_chops = [[_y] for _y in y]
                else:
                    for y_chop, _y in zip(y_chops, y): y_chop.append(_y)

        h *= scale
        w *= scale
        top = slice(0, h // 2)
        bottom = slice(h - h // 2, h)
        bottom_r = slice(h // 2 - h, None)
        left = slice(0, w // 2)
        right = slice(w - w // 2, w)

        if w % 2 != 0:
            right_r = slice(w // 2 - w + 1, None)
            bottom_r = slice(h // 2 - h + 1, None)
        else:
            right_r = slice(w // 2 - w, None)
            bottom_r = slice(h // 2 - h, None)

        # batch size, number of color channels
        b, c = y_chops[0][0].size()[:-2]
        y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
        for y_chop, _y in zip(y_chops, y):
            _y[..., top, left] = y_chop[0][..., top, left]
            _y[..., top, right] = y_chop[1][..., top, right_r]
            _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
            _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]

        if len(y) == 1: y = y[0]

        return y


@regist_model
class LGBPN(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9, pattern='baseline', group=1, head_ch=None, br2_blc=6, SIDD=True):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        if head_ch is None:
            head_ch = base_ch

        ly = []
        ly += [nn.Conv2d(in_ch, head_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        self.branch1 = local_branch(2, base_ch, num_module, group=group, head_ch=head_ch, SIDD=SIDD)
        self.branch2 = global_branch(stride=3, num_module=[br2_blc], in_ch=base_ch, head_ch=head_ch, SIDD=SIDD)

        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch // 2, out_ch, kernel_size=1)]
        self.tail = nn.Sequential(*ly)


    def forward(self, x, refine=False, dict=None):
        pad = 0
        b, c, h, w = x.shape

        x = self.head(x)
        br1 = self.branch1(x, refine, dict=dict)
        br2 = self.branch2(x, refine, dict=dict)

        x = torch.cat([br1, br2], dim=1)
        if pad != 0:
            x = x[:, :, :-pad, :-pad]

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        self.body = nn.Sequential(*ly)

    def forward(self, x):
        return x + self.body(x)

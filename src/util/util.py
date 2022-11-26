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



class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, ratio=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # self.zero_padding = nn.ZeroPad2d(padding)
        self.zero_padding = nn.ReflectionPad2d(padding)
        # self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        # nn.init.constant_(self.p_conv.weight, 0)
        # self.p_conv.register_backward_hook(self._set_lr)

        self.ratio = ratio

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        # offset = self.p_conv(x)
        b, c, h, w = x.shape
        offset = torch.zeros((b, 2*self.kernel_size**2, h, w), device=x.device)
        # (b, 2N, h, w)：对于每个位置，对应的卷积核大小N都包含x、y两个坐标
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)
        # 通过学习到每个点的相对于卷积核中心offset，生成相应的绝对坐标

        ########
        ## 直接通过torch.meshgrid得到11*11的网格，然后向中心缩放到5/11的大小，再加回去
        ########

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        # 得到偏移后绝对坐标后，通过floor得到左上角处的坐标
        q_rb = q_lt + 1
        # 得到右下角的坐标

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        # 对左上角和右下角的坐标范围进行clamp，都在feature map的大小中
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # 根据左上角和右下角，得到右上角和左下角的坐标

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)
        # 学习到的offset出去feature map之后直接clamp

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))  # 1+x0-x * 1+y0-y
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # ：N是x坐标；  N：是y坐标

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        # 经过offset之后，得到每个位置偏移之后的对应输入

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        # out = self.conv(x_offset)

        # return out
        return x_offset

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)
        # 每个卷积核的大小为N，生成kernel size * kernel size大小的相对位移，一共有x、y两个方向

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
            # torch.arange(self.kernel_size//2, w*self.stride+1-self.kernel_size//2+1, self.stride),
            # torch.arange(self.kernel_size//2, w*self.stride+1-self.kernel_size//2+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        # 每个卷积核的中心位置。与输入的x大小并不一定一样（当有dilation的时候，间隔出现p0）
        p_0 = p_0 - 1 + self.kernel_size // 2

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # h -= (self.kernel_size//2*2)
        # w -= (self.kernel_size//2*2)
        # offset对应的是所有p0拼起来之后的大小（128*128），不是原图的大小
        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # p = p_0 + p_n + offset
        # p = p_0 + (p_n / 5 * 2)


        # shift
        # # p_n = torch.maximum(p_n-0.1, torch.zeros_like(p_n))
        # shift = 0.5
        # p_n_zero = torch.where(p_n == 0)
        # p_n_big = torch.where(p_n > 0)
        # p_n_small = torch.where(p_n < 0)
        # p_n[p_n_big] -= shift
        # p_n[p_n_small] += shift


        p = p_0 + p_n * self.ratio
        # p = p_0 + p_n
        # 每个卷积核的中心位置 + 相对位置 + 学习到的offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
        
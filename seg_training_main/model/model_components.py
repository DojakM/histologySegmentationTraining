import math

import torch
import torch.nn as nn
from torch.nn import init


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(UnetConv, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)))

        else:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True)))

    def forward(self, inputs):
        conv = self.conv
        x = conv(inputs)
        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_size, out_size, False)
        self.up = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1)
        )

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        return self.conv(outputs0)

    def print(self, *kwargs):
        print(kwargs)

class DACBlock(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass


class resMultiKernerlPoolingBlock(nn.Module):
    def __init__(self, in_size):
        super(resMultiKernerlPoolingBlock, self).__init__()
    def forward(self):
        pass

def _upsample_like(x, size):
    return nn.Upsample(size=size, mode='nearest')(x)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes
class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super(RSU, self).__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module('rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module('rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.dropout_1 = nn.Dropout2d(0.0)
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(self.dropout_1(x))))

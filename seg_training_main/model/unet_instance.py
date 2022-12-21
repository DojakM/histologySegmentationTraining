from typing import Any

import torch.nn as nn
from torch.nn import init

from seg_training_main.model.model_components import UnetConv, UnetUp
from seg_training_main.model.unet_super import UnetSuper

def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Unet(UnetSuper):
    def __init__(self, num_classes, len_test_set, hparams, input_channels, is_deconv=True,
                 is_batchnorm=True, **kwargs):
        super().__init__(num_classes=num_classes, len_test_set=len_test_set, hparams = hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [32, 64, 128, 256]
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = UnetConv(self.in_channels, filters[0], True)
        self.conv2 = UnetConv(filters[0], filters[1], True)
        self.conv3 = UnetConv(filters[1], filters[2], True)
        self.center = UnetConv(filters[2], filters[3], True)
        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], True)
        self.up_concat2 = UnetUp(filters[2], filters[1], True)
        self.up_concat1 = UnetUp(filters[1], filters[0], True)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*256*256
        maxpool1 = self.maxpool(conv1)  # 16*128*128

        conv2 = self.conv2(maxpool1)  # 32*128*128
        maxpool2 = self.maxpool(conv2)  # 32*64*64

        conv3 = self.conv3(maxpool2)  # 64*64*64
        maxpool3 = self.maxpool(conv3)  # 64*32*32

        center = self.center(maxpool3)  # 256*16*16
        up3 = self.up_concat3(center, conv3)  # 64*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*256*256

        final = self.final(up1)

        return final

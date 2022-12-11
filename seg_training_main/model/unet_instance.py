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
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, num_classes, len_test_set, hparams, input_channels, min_filter, feature_scale=2, is_deconv=True,
                 is_batchnorm=True, **kwargs):
        super().__init__(num_classes=num_classes, len_test_set=len_test_set, hparams = hparams, **kwargs)

        self.in_channels = input_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [input_channels, 64, 128, 256, 512]
        self.convs = []
        self.concats = []
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        for i in range(0, len(filters) - 1):
            print(filters[i])
            self.convs.append(UnetConv(filters[i], filters[i + 1]))
            self.concats.append(UnetUp(filters[4 - i], filters[3 - i]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, *args, **kwargs):
        input = self.input
        for conv in self.convs:
            result = conv(input)
            input = self.maxpool(result)
        out = self.center(input)
        for iterator in range(0, len(self.concats)):
            out = self.concats[4 - iterator](out, self.convs[4 - iterator])
        final = self.final(out)
        return final

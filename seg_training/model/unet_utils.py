from torch.nn import *
import torch
import math
def init_weights(net, init_type='normal'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
def weights_init_kaiming(child):
    classname = child.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(child.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(child.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(child.weight.data, 1.0, 0.02)
        init.constant_(child.bias.data, 0.0)

class unetConv2(Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = Sequential(Conv2d(out_size, kernel_size, self.stride, self.padding),
                                     BatchNorm2d(out_size), ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
        else:
            for i in range(1, n + 1):
                conv = Sequential(Conv2d(in_size, out_size, kernel_size, self.stride, self.padding),
                                     ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        for child in self.children():
            init_weights(child, init_type = 'kaiming')

    def forward(self):
        return self

class unetUp(Module): # Upwards of the Unet
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = Sequential(UpsamplingNearest2d(scale_factor=2), Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for child in self.children():
            if child.__class__.__name__.find('unetConv2') != -1:    # What does this do?
                continue
            init_weights(child, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

def _upsample_like(x, size):   # Why not public?
    return Upsample(size, mode="Nearest")(x) # What happens hear exactly? Why nearest?

def _size_map(x, height):
    size = list(x.shape[-2:])   # What is this
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes
def set_dropout():
    return None

class REBNCONV(Module):
    def __init__(self):
        return None

    def forward(self):
        return None

class RSU(Module):
    def __init__(self):
        return None

    def forward(self):
        return None

    def _make_layers(self):
        return None


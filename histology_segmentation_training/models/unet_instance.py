import torch
import torch.nn as nn
from models.model_components import *
from models.unet_super import UnetSuper
from utils import weights_init
import sys

def __getattr__(name):
    return getattr(sys.modules[__name__], name)
class Unet(UnetSuper):
    """Unet

    Basic Unet which is used for medical image segmentation and classification
    original paper: https://arxiv.org/pdf/1505.04597
    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__( hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [16, 32, 64, 128]
        self.conv1 = UnetConv(self.in_channels, filters[0], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.center = UnetConv(filters[2], filters[3], is_batchnorm, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], kwargs["num_classes"], 1)
        if on_gpu:
            self.conv1.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs):
        maxpool = nn.MaxPool2d(kernel_size=2)
        conv1 = self.conv1(inputs)  # 16*256*256
        maxpool1 = maxpool(conv1)  # 16*128*128

        conv2 = self.conv2(maxpool1)  # 32*128*128
        maxpool2 = maxpool(conv2)  # 32*64*64

        conv3 = self.conv3(maxpool2)  # 64*64*64
        maxpool3 = maxpool(conv3)  # 64*32*32

        center = self.center(maxpool3)

        up3 = self.up_concat3(center, conv3)  # 64*64*64
        up2 = self.up_concat2(up3, conv2)  # 32*128*128
        up1 = self.up_concat1(up2, conv1)  # 16*256*256

        final = self.final(up1)
        finalize = nn.functional.softmax(final, dim=1)
        return finalize


    def print(self, args: torch.Tensor) -> None:
        print(args)

#### ==== model with spatial transformer ==== ####
class RTUnet(UnetSuper):
    """RTUnet

    A Unet with a spatial transformer network at the beginning
    Does not produce intended outcome
    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=True, on_gpu=False, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [8, 16, 32, 64]
        self.head1 = multiHeadBlock(2, input_channels, 1,  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd1 = forwardProcessingBlock(input_channels,  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv1 = UnetConv(input_channels, filters[0], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs[
            "dropout_val"])
        self.head2 = multiHeadBlock(2, filters[0], 2, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd2 = forwardProcessingBlock(filters[0],  gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv2 = UnetConv(filters[0], filters[1], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.head3 = multiHeadBlock(2, filters[1], 3, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.fwd3 = forwardProcessingBlock(filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.conv3 = UnetConv(filters[1], filters[2], is_batchnorm=True, gpus=on_gpu, dropout_val=kwargs["dropout_val"])


        # upsampling
        self.up_concat3 = UnetUp(filters[3], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat2 = UnetUp(filters[2], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.up_concat1 = UnetUp(filters[1], filters[0], gpus=on_gpu, dropout_val=kwargs["dropout_val"])

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], 7, 1)
        if on_gpu:
            self.head1.cuda()
            self.head2.cuda()
            self.conv2.cuda()
            self.conv3.cuda()
            self.center.cuda()
            self.up_concat3.cuda()
            self.up_concat2.cuda()
            self.up_concat1.cuda()
            self.final.cuda()
        self.apply(weights_init)

    def forward(self, inputs: torch.Tensor):
        maxpool = nn.MaxPool2d(kernel_size=2)
        x_s = torch.chunk(inputs, 8, 2)
        along_x = []
        for i in x_s:
            chunks = torch.chunk(i, 8, 3)
            along_x.append(chunks)
        merge_x = []
        for chunks in along_x:
            merge_y = []
            for chunk in chunks:
                x1 = self.head1(chunk)  # 16*64*64
                y1 = self.fwd1(x1)
                z1 = self.conv1(y1)
                maxpool1 = maxpool(z1)  # 16*32*32

                x2 = self.head2(maxpool1)  # 16*64*64
                y2 = self.fwd2(x2)
                z2 = self.conv2(y2)
                maxpool2 = maxpool(z2)  # 32*16*16

                x3 = self.head3(maxpool2)  # 16*64*64
                y3 = self.fwd3(x3)
                z3 = self.conv3(y3)

                up2 = self.up_concat2(z3, z2)  # 32*32*32
                up1 = self.up_concat1(up2, z1)  # 16*64*64

                final = self.final(up1)
                finalize = nn.functional.softmax(final, dim=1)
                merge_y.append(finalize)
            merge_x.append(torch.cat(merge_y, 3))
        return torch.cat(merge_x, 2)



#### ==== Context Unet ==== ####
class ContextUnet(UnetSuper):
    """Context Unet is a U-Net with added context modules and localization modules and a different way of generating
    the higher dimension feature maps. Additionally deep_supervision elements are present, however not meaningfully
    better

    """
    def __init__(self, hparams, input_channels, is_deconv=True, is_batchnorm=True, on_gpu=False,
                 deep_supervision=True, **kwargs):
        super().__init__(hparams=hparams, **kwargs)
        self.deep_supervision = deep_supervision
        self.in_channels = input_channels
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.input = input_channels
        filters = [16, 32, 64, 128, 256]
        self.conv1 = SimpleUnetConv(self.in_channels, filters[0], stride=1, gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context1 = ContextModule(filters[0], filters[0],gpus=on_gpu)
        self.ttt2 = SimpleUnetConv(filters[0], filters[1], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context2 = ContextModule(filters[1], filters[1], gpus=on_gpu)
        self.ttt3 = SimpleUnetConv(filters[1], filters[2], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context3 = ContextModule(filters[2], filters[2],  gpus=on_gpu)
        self.ttt4 = SimpleUnetConv(filters[2], filters[3], gpus=on_gpu, dropout_val=kwargs["dropout_val"])
        self.context4 = ContextModule(filters[3], filters[3], gpus=on_gpu)
        self.up_center = SimpleUnetUp(filters[3], filters[2], gpus=on_gpu)
        self.local1 = Localization(filters[3], filters[2], gpus=on_gpu)
        self.up1 = SimpleUnetUp(filters[2], filters[1], gpus=on_gpu)
        self.local2 = Localization(filters[2], filters[1], gpus=on_gpu)
        self.up2 = SimpleUnetUp(filters[1], filters[0], gpus=on_gpu)
        self.final = nn.Conv2d(filters[1], kwargs["num_classes"], 1)
        self.seg = SegmentationLayer(64, 32, 7, gpus=on_gpu)
        self.apply(weights_init)
        if on_gpu:
            self.conv1.cuda()
            self.context1.cuda()
            self.context2.cuda()
            self.context3.cuda()
            self.context4.cuda()
            self.ttt2.cuda()
            self.ttt3.cuda()
            self.ttt4.cuda()
            self.up_center.cuda()
            self.local1.cuda()
            self.local2.cuda()
            self.up1.cuda()
            self.up2.cuda()
            self.final.cuda()

    def forward(self, x):
        con1 = self.conv1(x) # 16*256*256
        son1 = self.context1(con1)
        plus1 = con1 + son1

        con2 = self.ttt2(plus1) # 32*128*128
        son2 = self.context2(con2)
        plus2 = con2 + son2

        con3 = self.ttt3(plus2) # 64*64*64
        son3 = self.context3(con3)
        plus3 = con3+son3

        con4 = self.ttt4(plus3) # 128*32*32
        son4 = self.context4(con4)
        plus4 = con4 + son4

        up_center = self.up_center(plus4) #64*64*64

        comb = torch.cat([plus3, up_center], dim=1) #128*64*64
        local1 = self.local1(comb) #64*64*64
        up1 = self.up1(local1) #32*128*128

        comb = torch.cat([plus2, up1], dim=1) #64*128*128
        local2 = self.local2(comb) #32*128*128
        up2 = self.up2(local2) #16*256*256

        comb = torch.cat([plus1, up2], dim=1) #32*256*256
        final = self.final(comb)  #7*256*256

        if self.deep_supervision:
            final = self.seg(local1, local2, final)

        return nn.functional.softmax(final, dim=1) #1*256*256

from model_components import *
import pytorch_lightning as pl
import torch.nn as nn


class Unet(pl.LightningModule):
    def __init__(self, num_classes, input):
        super().__init__()
        self.input = input
        filters = [input, 64, 128, 256, 512]
        self.convs = []
        self.concats = []
        self.class_weights = [0.0001, 1, 1, 1, 1, 1, 1]
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        for i in range(0, len(filters) - 1):
            print(filters[i])
            self.convs.append(UnetConv(filters[i], filters[i + 1]))
            self.concats.append(UnetUp(filters[4 - i], filters[3 - i]))

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


if __name__ == '__main__':
    Unet(7, 3)

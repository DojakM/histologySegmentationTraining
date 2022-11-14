import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_super import UnetSuper

def create_unet():
    return Unet()

class Unet(UnetSuper):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(self.dropout1(x), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

    def log_weights(self, step, writer):
        writer.add_histogram('weights/conv1/weight', self.conv1.weight.data, step)
        writer.add_histogram('weights/conv1/bias', self.conv1.bias.data, step)
        writer.add_histogram('weights/conv2/weight', self.conv2.weight.data, step)
        writer.add_histogram('weights/conv2/bias', self.conv2.bias.data, step)
        writer.add_histogram('weights/fc1/weight', self.fc1.weight.data, step)
        writer.add_histogram('weights/fc1/bias', self.fc1.bias.data, step)
        writer.add_histogram('weights/fc2/weight', self.fc2.weight.data, step)
        writer.add_histogram('weights/fc2/bias', self.fc2.bias.data, step)



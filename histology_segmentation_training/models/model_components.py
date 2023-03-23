import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UnetConv(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1, gpus=False, dropout_val=0.001):
        super(UnetConv, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.BatchNorm2d(out_size),
                nn.ReLU(inplace=True)))
        else:
            self.conv = nn.Sequential(nn.Sequential(
                nn.Dropout(dropout_val),
                nn.Conv2d(in_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_val),
                nn.Conv2d(out_size, out_size, ks, padding=padding, stride=stride),
                nn.ReLU(inplace=True)))
        if gpus:
            self.conv.cuda()

    def forward(self, inputs):
        conv = self.conv
        x = conv(inputs)
        return x


class UnetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, gpus=False, dropout_val=0):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_size, out_size, False)
        self.up = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_size, out_size, 1)
        )
        if gpus:
            self.conv.cuda()
            self.up.cuda()

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], dim=1)
        return self.conv(outputs0)

#### ==== Spatial Transformer kinda doesn't do anything ==== ####
class SPTTo(nn.Module):
    def __init__(self, in_size, out_size, stride=1, ks=3, dropout_val=0, gpus=False):
        super(SPTTo, self).__init__()

        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, ks, padding=1, stride=stride),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))

        # spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),  # 256*256*3
            nn.MaxPool2d(2, stride=2),  # 250*250*8
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),  # 125*125*8
            nn.MaxPool2d(2, stride=2),  # 121*121*16
            nn.ReLU(True)  # 60*60*16
        )
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(1024, 16),
            nn.Linear(16, 3 * 2)
        )
        self.fc_loc[1].weight.data.zero_()
        self.fc_loc[1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1) * xs.size(2) * xs.size(3))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, theta

    def rev_stn(self, x, theta: torch.tensor):
        empty_tensor = torch.empty(theta.size())
        for val, batch in enumerate(theta):
            new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]])), 0)
            theta = new_theta.inverse()
            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        x, theta = self.stn(x)
        x = self.conv(x)
        return x, theta

class SPTBack(nn.Module):
    def __init__(self, in_size, out_size):
        super(SPTBack, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, 1)

    def rev_stn(self, x, theta: torch.tensor):
        empty_tensor = torch.empty(theta.size())
        for val, batch in enumerate(theta):
            new_theta = torch.cat((batch, torch.tensor([[0, 0, 1]])), 0)
            theta = new_theta.inverse()
            theta = theta[:2, :]
            empty_tensor[val, :, :] = theta
        grid = F.affine_grid(empty_tensor, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x, theta):
        x = self.conv(x)
        self.rev_stn(x, theta)
        return x


#### ==== In Testing ==== ####
class Context(nn.Module):
    def __init__(self, in_size, out_size, gpus=False):
        super(Context, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, out_size, 3),
            nn.Dropout2d(p=0.3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, 3)
        )
    def forward(self, x):
        x = self.conv1(x)
        return x

class Localization(nn.Module):
    def __init__(self, in_size, out_size, dropout_val = 0):
        super(Localization, self).__init__()
        self.conv = nn.Sequential(nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Conv2d(in_size, out_size, kernel_size=3),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_val),
            nn.Conv2d(out_size, out_size, kernel_size=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)))
    def forward(self, x):
        self.conv(x)
        return x

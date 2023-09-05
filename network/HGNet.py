import torch
import torch.nn as nn
import torch.nn.functional as F

from .UNet import UNet


class HGNet(nn.Module):

    def __init__(self, n_stacks=2, in_channels=1, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(HGNet, self).__init__()
        self.n_stacks = n_stacks
        HGs = [UNet(in_channels, n_classes, feature_scale, is_deconv, is_batchnorm)]
        out_channels = HGs[0].out_channels
        for n in range(1, n_stacks):
            HGs.append(UNet(out_channels, n_classes, feature_scale, is_deconv, is_batchnorm))
        self.HGs = nn.ModuleList(HGs)
        self.remap_input = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                         nn.BatchNorm2d(out_channels),
                                         nn.ReLU(inplace=True))
        self.remap_output = nn.Sequential(nn.Conv2d(n_classes, out_channels, 1),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU(inplace=True))
        self.remap_feature = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1),
                                           nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True))

    def forward(self, x):
        y, f = self.HGs[0](x)
        Y, F = y[..., None], f[..., None]

        x = self.remap_input(x)
        for n in range(1, self.n_stacks):
            x = x + self.remap_feature(f) + self.remap_output(y)
            y, f = self.HGs[n](x)
            Y = torch.cat((Y, y[..., None]), dim=-1)
            F = torch.cat((F, f[..., None]), dim=-1)

        return Y, F


class HGMSELoss(nn.Module):
    def __init__(self, n_stacks, weights=None):
        super(HGMSELoss, self).__init__()
        if weights is None:
            weights = [1. / 2. ** (n_stacks - 1 - i) for i in range(n_stacks)]
            self.weights = [w / sum(weights) for w in weights]
        else:
            self.weights = weights
        print("Weights for the hourglasses: {}".format(self.weights))

    def forward(self, predict, target):
        loss = 0.
        for i in range(predict.shape[-1]):
            loss += self.weights[i] * F.mse_loss(predict[..., i], target)

        return loss

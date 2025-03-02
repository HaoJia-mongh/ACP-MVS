import torch
import torch.nn as nn
import sys
sys.path.append("..")


class ContextAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(ContextAtt, self).__init__()

        self.context_att = nn.Sequential(
            L_2DCNN(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1))

    def forward(self, init_weight, context_feat):
        cnet_att = self.context_att(context_feat).unsqueeze(2)
        consti_weight = torch.sigmoid(cnet_att)*init_weight
        return consti_weight


class L_3DCNN(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, relu=True, **kwargs):
        super(L_3DCNN, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x


class L_2DCNN(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, relu=True, **kwargs):
        super(L_2DCNN, self).__init__()

        self.relu = relu
        self.use_bn = bn
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)
        return x
import torch.nn as nn
import torch

# Selfattention implementation code borrowed from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class SABlock(nn.Module):

    def __init__(self, dim_in, activation):

        super(SABlock, self).__init__()

        self.channel_in = dim_in
        self.activation = activation

        self.query_conv = nn.Conv2d(
            dim_in, dim_in//8, kernel_size=1)
        self.key_conv = nn.Conv2d(dim_in, dim_in//8, kernel_size=1)

        self.value_conv = nn.Conv2d(
            in_channels=dim_in, out_channels=dim_in, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * H * W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        batchsize, channel, height, width = x.size()

        proj_query = self.query_conv(x).view(batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(
            batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batchsize, -1, width*height).permute(0, 2, 1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batchsize, -1, height, width)

        return out, attention


class ResidualBlock(nn.Module):

    def __init__(self, dim_in):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):

        identity = x

        output = self.conv1(output)
        output = self.conv2(output)

        return nn.ReLU(output + identity)


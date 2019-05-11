import torch.nn as nn
from block import *

class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.sablock1 = SABlock(dim_in=256, activation='relu')

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.sablock2 = SABlock(dim_in=512, activation='relu')

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(1024*4*4, 1024)

        self.fc2 = nn.Linear(1024, 1024*4*4)

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=4096,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.upscaleblock = nn.PixelShuffle(2)

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x, _ = sablock1(x)

        x = self.conv4(x)

        x, _ = self.sablock2(x)

        x = fc1(x)

        x = fc2(x)

        x = x.view(-1, 1024, 4, 4)

        x = self.conv6(x)

        x = upscaleblock(x)

        return x


class Decoder(nn.module):

    def __init__(self, dim_in=512):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=256 *
                      2*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock1 = nn.PixelShuffle(2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128*2 *
                      2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock2 = nn.PixelShuffle(2)

        self.sablock1 = SABlock(dim_in=128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64*2*2,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.upscaleblock3 = nn.PixelShuffle(2)

        self.resblock = ResidualBlock(dim_in=64)

        self.sablock1 = SABlock(dim_in=64, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3,
                      kernel_size=5, stride=1, padding=0),
            nn.tanh(),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.upscaleblock1(x)

        x = self.conv2(x)
        x = self.upscaleblock2(x)

        x, _ = self.sablock1(x)

        x = self.conv3(x)
        x = self.upscaleblock3(x)

        x = self.resblock(x)

        x, _ = self.sablock2(x)

        mask = self.conv4(x)

        output = self.conv5(x)

        return output, mask


class Discriminator(nn.Module):

    def __init__(self, dim_in):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim_in, out_channels=64,
                      kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.sablock1 = SABlock(128, activation=None)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
        )

        self.sablock2 = SABlock(256, activation=None)

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1,
                      kernel_size=5, stride=1, padding=0),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x, _ = self.sablock1(x)
        x = self.conv3(x)

        x, _ = self.sablock2(x)
        x = self.conv4(x)

        return x

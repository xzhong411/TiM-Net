import torch
import torch.nn as nn
import torch.nn.functional as F
from core.unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.pool = nn.MaxPool2d((2, 2))
        self.conv1 = DoubleConv(n_channels, 32)
        self.conv2 = Conv(n_channels, 64)
        self.conv3 = Conv(n_channels, 128)
        self.conv4 = Conv(n_channels, 256)
        self.down1 = Down_Cat(32, 64)
        self.down2 = Down_Cat(64, 128)
        self.down3 = Down_Cat(128, 256)
        self.down4 = Down(256, 512)
        self.transformer2 = Embeddings(768, 256, 64)
        # self.transformer3 = Embeddings(768, 128, 128)
        # self.transformer4 = Embeddings(768, 64, 256)

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.out6 = OutConv(256, n_classes)
        self.out7 = OutConv(128, n_classes)
        self.out8 = OutConv(64, n_classes)
        self.out9 = OutConv1(32, n_classes)

    def forward(self, x):
        p2 = self.pool(x)
        p3 = self.pool(p2)
        p4 = self.pool(p3)
        p2c = self.conv2(p2)
        p3c = self.conv3(p3)
        p4c = self.conv4(p4)
        x1 = self.conv1(x)
        x2 = self.down1(x1, p2c)

        x2_1 = self.transformer2(x2)

        x3 = self.down2(x2, p3c)

        x4 = self.down3(x3, p4c)

        x5 = self.down4(x4)

        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2_1)
        x9 = self.up4(x8, x1)

        out6 = self.out6(x6)
        out7 = self.out7(x7)
        out8 = self.out8(x8)
        out9 = self.out9(x9)
        # out10 = (out6 + out7 + out8 + out9) / 4
        out10 = out6 * 0.1 + out7 * 0.25 + out8 * 0.25 + out9 * 0.4
        return [out10, out6, out7, out8, out9]
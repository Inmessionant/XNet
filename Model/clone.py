import torch
from torch import nn


class XNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(XNet, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pooling = MaxPool2dStaticSamePadding(3, 2)

        self.conv = conv(in_channels=in_channels, out_channels=32)

        self.m11 = BasicBlock(32, 32, 64)
        self.m12 = BasicBlock(64, 64, 128)
        self.m13 = BasicBlock(128, 128, 256)
        self.m14 = BasicBlock(256, 256, 512)
        self.m15 = BasicBlock(512, 512, 512)

        self.m24 = BasicBlock(1024, 512, 256)
        self.m23 = BasicBlock(512, 256, 128)
        self.m22 = BasicBlock(256, 128, 64)

        self.m31 = BasicBlock(128, 64, 128)
        self.m32 = BasicBlock(320, 160, 320)
        self.m33 = BasicBlock(704, 352, 704)
        self.m34 = BasicBlock(1472, 736, 1472)
        self.m35 = BasicBlock(1984, 992, 1984)

        self.conv1 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(320, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(704, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(1472, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(1984, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.fusion = nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv1 = self.conv(x)

        x11 = self.m11(conv1)
        x11B = x11

        x12 = self.m12(self.pooling(x11))
        x12A, x12B = x12, x12

        x13 = self.m13(self.pooling(x12))
        x13A, x13B = x13, x13

        x14 = self.m14(self.pooling(x13))
        x14A, x14B = x14, x14

        x15 = self.m15(self.pooling(x14))
        x15B = x15

        x24 = self.m24(torch.cat((x14A, self.upsample(x15)), 1))
        x24A = x24

        x23 = self.m23(torch.cat((x13A, self.upsample(x24)), 1))
        x23A = x23

        x22 = self.m22(torch.cat((x12A, self.upsample(x23)), 1))
        x22A = x22

        x31 = self.m31(torch.cat((x11B, self.upsample(x22)), 1))

        x32 = self.m32(torch.cat((x22A, x12B, self.pooling(x31)), 1))

        x33 = self.m33(torch.cat((x23A, x13B, self.pooling(x32)), 1))

        x34 = self.m34(torch.cat((x24A, x14B, self.pooling(x33)), 1))

        x35 = self.m35(torch.cat((x15B, self.pooling(x34)), 1))

        out1 = self.conv1(x31)

        out2 = _upsample_like(self.conv2(x32), out1)

        out3 = _upsample_like(self.conv2(x33), out1)

        out4 = _upsample_like(self.conv2(x34), out1)

        out5 = _upsample_like(self.conv2(x35), out1)

        fusion = self.fusion(torch.cat((out1, out2, out3, out4, out5), 1))

        return F.sigmoid(fusion)


m = nn.Conv2d(2, 5, 1)
input = torch.randn(1, 2, 35, 256)
output = m(input)
print(output.shape)
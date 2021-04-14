import torch
from torchviz import make_dot
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from Model.torch_utils import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):  # 保证gn中能整除16，实验证明一组16通道比较好
        super(conv, self).__init__()
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1,
                                                      groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.gn = nn.GroupNorm(int(out_channels / 16), out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.gelu(self.gn(x))

        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.pointwise_conv_in2mid = Conv2dStaticSamePadding(in_channels, mid_channels, kernel_size=1, stride=1)
        self.pointwise_conv_mid2out = Conv2dStaticSamePadding(mid_channels, out_channels, kernel_size=1, stride=1)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.pooling = MaxPool2dStaticSamePadding(3, 2)
        self.conv_1 = conv(1 * mid_channels, mid_channels)
        self.conv_2 = conv(2 * mid_channels, mid_channels)
        self.conv_3 = conv(3 * mid_channels, mid_channels)
        self.se = SELayer(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.pointwise_conv_in2mid(x)

        p3_0 = self.conv_1(x)
        p3_0b = p3_0
        p4_0 = self.conv_1(self.pooling(p3_0))
        p4_0a, p4_0b = p4_0, p4_0
        p5_0 = self.conv_1(self.pooling(p4_0))
        p5_0a, p5_0b = p5_0, p5_0
        p6_0 = self.conv_1(self.pooling(p5_0))
        p6_0a, p6_0b = p6_0, p6_0
        p7_0 = self.conv_1(self.pooling(p6_0))
        p7_0b = p7_0

        p6_1 = self.conv_2(torch.cat((p6_0a, self.upsample(p7_0)), 1))
        p6_1a = p6_1
        p5_1 = self.conv_2(torch.cat((p5_0a, self.upsample(p6_1)), 1))
        p5_1a = p5_1
        p4_1 = self.conv_2(torch.cat((p4_0a, self.upsample(p5_1)), 1))
        p4_1a = p4_1

        p7_2 = self.conv_1(p7_0b)
        p6_2 = self.conv_3(torch.cat((p6_0b, p6_1a, self.upsample(p7_2)), 1))
        p5_2 = self.conv_3(torch.cat((p5_0b, p5_1a, self.upsample(p6_2)), 1))
        p4_2 = self.conv_3(torch.cat((p4_0b, p4_1a, self.upsample(p5_2)), 1))
        p3_2 = self.conv_3(torch.cat((p3_0b, self.upsample(p4_1), self.upsample(p4_2)), 1))

        x = self.pointwise_conv_mid2out(p3_2)
        x = self.se(x)

        return x


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
        print(x11.shape)

        x12 = self.m12(self.pooling(x11))
        x12A, x12B = x12, x12
        print(x12.shape)

        x13 = self.m13(self.pooling(x12))
        x13A, x13B = x13, x13
        print(x13.shape)
        print(self.pooling(x13).shape)

        x14 = self.m14(self.pooling(x13))
        x14A, x14B = x14, x14
        print(x14.shape)

        x15 = self.m15(self.pooling(x14))
        x15B = x15

        x24 = self.m24(torch.cat((x14B, self.upsample(x15)), 1))
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


# Bug ,BasicBlock最小支持输入尺寸为80，所以得改XNet网络深度

# torchsummary
model = BasicBlock(128, 128, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(summary(model, (128, 80, 80)))


# # torchstat
# model = XNet()
# print(stat(model, (3, 320, 320)))


# pytorchviz
model = BasicBlock(128, 128, 256)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

x = torch.randn(1, 128, 80, 80).requires_grad_(True)
y = model(x)
vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vis_graph.view()
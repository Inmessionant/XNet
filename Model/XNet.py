import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from Model.torch_utils import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding


# input = torch.randn(2, 2, 10, 10)
# m = nn.UpsamplingBilinear2d(scale_factor=2)
# output = m(input)
# print(output.shape)


# nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)  降采样
# nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)  1x1conv
# nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, dilation=1, bias=False) 不改变原尺寸


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

    def forward(self, x):
        hx = x
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

        return x


class XNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super(XNet, self).__init__()

    def forward(self, x):
        return


# torchsummary
# model = BasicBlock(in_channels=64, mid_channels=32, out_channels=32)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# print(summary(model, (64, 320, 320)))

# # torchstat
# model = XNet()
# print(stat(model, (3, 320, 320)))

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
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.pooling = MaxPool2dStaticSamePadding(3, 2)

        self.conv1 = conv(in_channels=in_channels, out_channels=32)

        self.x11 = BasicBlock(32, 32, 64)
        self.x12 = BasicBlock(64, 64, 128)
        self.x13 = BasicBlock(128, 128, 256)

        self.x22pre = nn.Conv2d(384, 256, kernel_size=1)
        self.x22 = BasicBlock(256, 256, 256)

        self.x31pre = nn.Conv2d(320, 256, kernel_size=1)
        self.x31 = BasicBlock(256, 256, 256)

        self.x32pre = nn.Conv2d(640, 256, kernel_size=1)
        self.x32 = BasicBlock(256, 256, 256)

        self.x33pre = nn.Conv2d(512, 256, kernel_size=1)
        self.x33 = BasicBlock(256, 256, 256)

        self.x42pre = nn.Conv2d(512, 256, kernel_size=1)
        self.x42 = BasicBlock(256, 256, 256)

        self.x51pre = nn.Conv2d(512, 32, kernel_size=1)
        self.x51 = BasicBlock(32, 32, 64)

        self.x52pre = nn.Conv2d(576, 64, kernel_size=1)
        self.x52 = BasicBlock(64, 64, 128)

        self.x53pre = nn.Conv2d(384, 128, kernel_size=1)
        self.x53 = BasicBlock(128, 128, 256)

        self.convx1 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.convx2 = nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.convx3 = nn.Conv2d(256, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.fusion = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv1 = self.conv1(x)

        x11 = self.x11(conv1)
        x11B = x11

        x12 = self.x12(self.pooling(x11))
        x12A, x12B = x12.clone(), x12.clone()

        x13 = self.x13(self.pooling(x12))
        x13B = x13.clone()

        x22pre = self.x22pre(torch.cat((x12A, self.upsample(x13)), 1))
        x22 = self.x22(x22pre)
        x22A = x22.clone()

        x31pre = self.x31pre(torch.cat((x11B, self.upsample(x22)), 1))
        x31 = self.x31(x31pre)
        x31B = x31.clone()

        x32pre = self.x32pre(torch.cat((x22A, x12B, self.pooling(x31)), 1))
        x32 = self.x32(x32pre)
        x32A, x32B = x32.clone(), x32.clone()
        print(x32.shape)
        print(x13.shape)

        x33pre = self.x33pre(torch.cat((x13B, self.pooling(x32)), 1))
        x33 = self.x33(x33pre)
        x33B = x33.clone()

        x42pre = self.x42pre(torch.cat((x32A, self.upsample(x33)), 1))
        x42 = self.x42(x42pre)
        x42A = x42.clone()

        x51pre = self.x51pre(torch.cat((x31B, self.upsample(x42)), 1))
        x51 = self.x51(x51pre)

        x52pre = self.x52pre(torch.cat((x42A, x32B, self.pooling(x51)), 1))
        x52 = self.x52(x52pre)

        x53pre = self.x53pre(torch.cat((x33B, self.pooling(x52)), 1))
        x53 = self.x53(x53pre)

        out1 = self.convx1(x51)
        out2 = self.upsample(self.convx2(x52))
        out3 = self.upsample2(self.convx3(x53))

        fusion = self.fusion(torch.cat((out1, out2, out3), 1))

        return F.sigmoid(fusion)


# torchsummary
model = XNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(summary(model, (3, 320, 320)))

# # torchstat
# model = XNet()
# print(stat(model, (3, 320, 320)))


# pytorchviz
# model = BasicBlock(128, 128, 256)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# x = torch.randn(1, 128, 80, 80).requires_grad_(True)
# y = model(x)
# vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# vis_graph.view()

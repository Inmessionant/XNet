import torch
import torch.nn as nn
from Model.torch_utils import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

# input = torch.randn(2, 2, 10, 10)
# m = nn.Conv2d(2, 5, kernel_size=1, bias=False)
# output = m(input)
# print(output.shape)


# nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=2, padding=1, bias=False, groups=inplanes)  降采样
# nn.Conv2d(inplanes, inplanes, kernel_size=1, bias=False)  1x1conv
# nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, dilation=1, bias=False) 不改变原尺寸

# self.gn = nn.GroupNorm(out_channels/16, out_channels)

# nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3)

# just pointwise_conv applies bias, depthwise_conv has no bias.

# nn.UpsamplingBilinear2d(scale_factor=2) 上采样


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):  # 保证gn中能整除16，实验证明一组16通道比较好
        super(conv, self).__init__()
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.gn = nn.GroupNorm(out_channels/16, out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.gelu(self.gn(x))
        return x


class XNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=1):
        super(XNet, self).__init__()

    def forward(self, x):
        return

# torchsummary
# model = XNet()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# print(summary(model, (3, 320, 320)))

# # torchstat
# model = XNet()
# print(stat(model, (3, 320, 320)))

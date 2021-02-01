import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
from torchsummary import summary


# GN使用，将6通道分成3组，一组2通道；实验证明一组16通道比较好；
input = torch.randn(20, 6, 10, 10)
 m = nn.GroupNorm(3, 6)
 output = m(input)


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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    
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

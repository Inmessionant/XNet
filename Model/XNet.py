import torch
import torch.nn as nn
import torch.nn.functional as F
from torchstat import stat
from torchsummary import summary


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

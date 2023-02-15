import torch
import torch.nn as nn
from primary_caps import PrimaryCaps
from conv_caps import ConvCaps


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 9)
        self.primary_caps = PrimaryCaps(256, 32, 9, 2, 0)
        self.digit_caps = ConvCaps(32, 10, 16, 1, 0, 32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.primary_caps(out)
        out = self.digit_caps(out)
        out_norm = torch.sqrt((out ** 2).sum(dim=2, keepdim=True))
        out = out_norm / (1 + out_norm) * out
        out = out.view(out.size(0), -1)
        return out

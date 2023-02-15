import torch
import torch.nn as nn

class PrimaryCaps(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PrimaryCaps, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size ** 2, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), self.out_channels, -1)
        out = out.transpose(1, 2)
        out = squash(out)
        return out


def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    out = scale * x / torch.sqrt(squared_norm)
    return out

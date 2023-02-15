import torch
import torch.nn as nn
from math import ceil

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            Swish()
        )

class SqueezeExcitation(nn.Module):
    def __init__(self, in_planes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, se_planes, kernel_size=1),
            Swish(),
            nn.Conv2d(se_planes, in_planes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, expand_ratio=1, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        hidden_planes = in_planes * expand_ratio
        se_planes = int(in_planes * se_ratio)
        self.use_se = (se_ratio is not None) and (0 < se_ratio <= 1)

        layers = []
        # expand
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_planes, hidden_planes, kernel_size=1))
        # depthwise conv
        layers.append(ConvBNReLU(hidden_planes, hidden_planes, kernel_size=kernel_size, stride=stride, groups=hidden_planes))
        # squeeze and excite
        if self.use_se:
            layers.append(SqueezeExcitation(hidden_planes, se_planes))
        # project
        layers.append(nn.Conv2d(hidden_planes, out_planes, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))

        self.conv = nn.Sequential(*layers)

        # skip connection
        self.has_skip = (stride == 1) and (in_planes == out_planes)
        if self.has_skip:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.has_skip:
            out = out + self.skip(x)
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_blocks, width_factor, depth_factor, dropout_rate, num_classes=10):
        super(EfficientNet, self).__init__()
        # according to the paper, we can choose alpha, beta, gamma values from 1.0 ~ 2.0
        # based on the input image size
        alpha, beta, gamma = 1.2, 1.1, 1.15
        base_planes = 32
        out_planes = int(ceil(base_planes * width_factor))

        self.conv1 = ConvBNReLU(3, out_planes, kernel_size=3, stride=2)
        in_planes = out_planes

        self.blocks = nn.Sequential()
        for i, num_block in enumerate(num_blocks):
            out_planes = int(ceil(base_planes * width_factor * gamma ** i))
            hidden_planes = int(in_planes * beta)
            stride = 1 if i == 0 else 2
            self.blocks.add_module(f"block{i}", self._make_layer(
                in_planes,
                out_planes,
                hidden_planes,
                num_block,
                stride=stride,
                se_ratio=0.25
            ))
            in_planes = out_planes

        # head
        self.head = nn.Sequential(
            ConvBNReLU(in_planes, in_planes * 4, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_planes * 4, num_classes)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.head(out)
        return out

    def _make_layer(self, in_planes, out_planes, hidden_planes, num_blocks, stride=1, se_ratio=0.25):
        layers = []
        # MBConv blocks
        layers.append(MBConvBlock(in_planes, out_planes, kernel_size=3, stride=stride, expand_ratio=1, se_ratio=se_ratio))
        for i in range(1, num_blocks):
            layers.append(MBConvBlock(out_planes, out_planes, kernel_size=3, stride=1, expand_ratio=hidden_planes/out_planes, se_ratio=se_ratio))
        return nn.Sequential(*layers)

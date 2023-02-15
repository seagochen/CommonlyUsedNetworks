import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)

    def forward(self, x):
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = F.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out.expand_as(x)
        return out


class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction_ratio=reduction_ratio)
        self.stride = stride

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class SEResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes=10, reduction_ratio=16):
        super(SEResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_layers[0], reduction_ratio=reduction_ratio)
        self.layer2 = self._make_layer(block, 128, num_layers[1], stride=2, reduction_ratio=reduction_ratio)
        self.layer3 = self._make_layer(block, 256, num_layers[2], stride=2, reduction_ratio=reduction_ratio)
        self.layer4 = self._make_layer(block, 512, num_layers[3], stride=2, reduction_ratio=reduction_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_layer(self, block, out_channels, num_blocks, stride=1, reduction_ratio=16):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, reduction_ratio))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, reduction_ratio=reduction_ratio))
        return nn.Sequential(*layers)

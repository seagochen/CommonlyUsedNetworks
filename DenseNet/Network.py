import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = nn.ReLU(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = nn.ReLU(out)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.bn(x)
        out = nn.ReLU(out)
        out = self.conv(out)
        out = self.pool(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, num_layers, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        num_blocks = len(num_layers)
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense_block(block, 2 * growth_rate, num_layers[0], growth_rate)
        in_channels = 2 * growth_rate + num_layers[0] * growth_rate
        out_channels = int(in_channels * reduction)
        self.trans1 = Transition(in_channels, out_channels)
        in_channels = out_channels
        for i in range(1, num_blocks):
            num_layer = num_layers[i]
            out_channels += num_layer * growth_rate
            block = self._make_dense_block(block, in_channels, num_layer, growth_rate)
            self.add_module(f'dense_block_{i + 1}', block)
            in_channels += num_layer * growth_rate
            if i != num_blocks - 1:
                out_channels = int(in_channels * reduction)
                trans = Transition(in_channels, out_channels)
                self.add_module(f'transition_{i + 1}', trans)
                in_channels = out_channels
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        out = self.trans1(out)
        for i in range(2, self.num_blocks + 1):
            dense_block = getattr(self, f'dense_block_{i}')
            out = dense_block(out)
            if i != self.num_blocks:
                trans = getattr(self, f'transition_{i}')
                out = trans(out)
        out = self.bn(out)
        out = nn.ReLU(out)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _make_dense_block(self, block, in_channels, num_layers, growth_rate):
        layers = []
        for i in range(num_layers):
            layers.append(block(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)


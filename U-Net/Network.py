import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.conv(x)
        return out
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down_conv1 = DoubleConv(in_channels, 64)
        self.down_conv2 = DoubleConv(64, 128)
        self.down_conv3 = DoubleConv(128, 256)
        self.down_conv4 = DoubleConv(256, 512)
        self.down_conv5 = DoubleConv(512, 1024)
        self.up_conv1 = UpConv(1024, 512)
        self.up_conv2 = UpConv(512, 256)
        self.up_conv3 = UpConv(256, 128)
        self.up_conv4 = UpConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = nn.MaxPool2d(2)(x1)
        x2 = self.down_conv2(x2)
        x3 = nn.MaxPool2d(2)(x2)
        x3 = self.down_conv3(x3)
        x4 = nn.MaxPool2d(2)(x3)
        x4 = self.down_conv4(x4)
        x5 = nn.MaxPool2d(2)(x4)
        x5 = self.down_conv5(x5)
        x = self.up_conv1(x5, x4)
        x = self.up_conv2(x, x3)
        x = self.up_conv3(x, x2)
        x = self.up_conv4(x, x1)
        out = self.out_conv(x)
        return out

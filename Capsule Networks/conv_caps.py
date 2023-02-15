import torch
import torch.nn as nn
from primary_caps import squash


class ConvCaps(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_routes, num_iterations):
        super(ConvCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_routes = num_routes
        self.num_iterations = num_iterations
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        batch_size = x.size(0)
        num_caps = x.size(1)
        H, W = x.size(2), x.size(3)
        num_routes = self.num_routes if self.num_routes != -1 else num_caps
        out_h = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        u_hat = torch.squeeze(torch.matmul(self.weight, x.view(batch_size, num_caps, H * W)), dim=-1)
        u_hat = u_hat.view(batch_size, num_caps, self.out_channels, out_h, out_w)
        u_hat = u_hat.permute(0, 3, 4, 1, 2).contiguous()
        b = torch.zeros(batch_size, out_h, out_w, self.out_channels, 1).to(x.device)

        for i in range(self.num_iterations):
            c = nn.functional.softmax(b, dim=1)
            c = c / (torch.sum(c, dim=3, keepdim=True) + 1e-10)

            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = squash(s)

            if i < self.num_iterations - 1:
                delta_b = (v * u_hat).sum(dim=-1, keepdim=True)
                b = b + delta_b

        v = torch.squeeze(v, dim=1)
        return v

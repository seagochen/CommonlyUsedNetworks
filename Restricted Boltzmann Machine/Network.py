import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible))
        self.v_bias = nn.Parameter(torch.randn(num_visible))
        self.h_bias = nn.Parameter(torch.randn(num_hidden))
        
    def forward(self, v):
        h = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h[h > torch.rand_like(h)] = 1.
        h[h <= torch.rand_like(h)] = 0.
        v = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        v[v > torch.rand_like(v)] = 1.
        v[v <= torch.rand_like(v)] = 0.
        return v, h

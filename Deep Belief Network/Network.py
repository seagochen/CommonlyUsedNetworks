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
        return v, h
    
class DBN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            if i == 0:
                num_visible = input_size
            else:
                num_visible = hidden_sizes[i - 1]
            rbm = RBM(num_visible, hidden_sizes[i])
            self.rbms.append(rbm)
        self.fc = nn.Linear(hidden_sizes[-1], 10)
        
    def forward(self, x):
        for rbm in self.rbms:
            x, _ = rbm(x)
        x = self.fc(x)
        return x

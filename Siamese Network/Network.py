import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, x1, x2):
        out1 = self.fc1(x1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)
        out1 = self.relu(out1)
        out2 = self.fc1(x2)
        out2 = self.relu(out2)
        out2 = self.fc2(out2)
        out2 = self.relu(out2)
        out = torch.abs(out1 - out2)
        out = self.fc3(out)
        return out

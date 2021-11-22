'''Fully Connected Network in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1   = nn.Linear(3*32*32, 1024)
        self.fc2   = nn.Linear(1024, 512)
        self.fc3   = nn.Linear(512, 256)
        self.fc4   = nn.Linear(256, 64)
        self.fc5   = nn.Linear(64, 10)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class pnuemoniaCNN(nn.Module):
    def __init__(self):
        super(pnuemoniaCNN, self).__init__()
        self.conv10 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=5, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=16 * 56 * 56, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.conv10(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv16(x)
        x = F.relu(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        return out

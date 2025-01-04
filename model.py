'''
model.py
Defining the creator-recognizing model.
DO NOT RUN THIS FILE DIRECTLY.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CreatorRecognizerFor4K(nn.Module):
    def __init__(self, num_classes=1):
        super(CreatorRecognizerFor4K, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(1, 5))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3))
        self.fc1 = nn.Linear(63360, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # [B, 3, 4, 1000]
        x = F.relu(self.conv1(x))  # [B, 16, 1, 996]
        x = F.relu(self.conv2(x))  # [B, 32, 1, 992]
        x = F.relu(self.conv3(x))  # [B, 64, 1, 990]
        x = torch.flatten(x, start_dim=1)  # [B, 63360]
        x = F.relu(self.fc1(x))  # [B, 128]
        x = self.fc2(x)  # [B, num_classes]
        return F.log_softmax(x, dim=1)
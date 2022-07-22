"""
"""
import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(config.SEED)

class DQN(nn.Module):
    def __init__(self, input_dims, action_dims, fc1_dims=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, action_dims)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)
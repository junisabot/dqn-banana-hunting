"""
"""
import random
import copy
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim

import config
from util.utils import ReplayBuffer
from network.dqn import DQN
torch.manual_seed(config.SEED)
random.seed(config.SEED)

device = torch.device(config.DEVICE)

class AGENT_DQN():

    def __init__(self, input_dims, action_dims):
        self.input_dims = input_dims
        self.action_dims = action_dims

        self.model = DQN(input_dims, action_dims).to(device)
        self.target_model = DQN(input_dims, action_dims).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)
        
        self.memory = ReplayBuffer()
        self.temporal_step = 0
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action_values = self.model(state)
        self.model.train()
                
        if np.random.uniform() < config.EPSILON:
            return random.choice(np.arange(self.action_dims))            
        else:
            action = np.argmax(action_values.cpu().data.numpy())
            return action
            
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):        
        states, actions, rewards, next_states, dones = experiences        

        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + config.GAMMA * Q_targets_next * (1 - dones)
        Q_value = self.model(states).gather(1, actions)
        loss = F.smooth_l1_loss(Q_value, Q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_model()

    def update_model(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(config.TAU*local_param.data + (1.0-config.TAU)*target_param.data)
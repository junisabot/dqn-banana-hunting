"""
"""
import torch
import random
import numpy as np
from collections import deque
from recordtype import recordtype

import config

device = torch.device(config.DEVICE)

class ReplayBuffer:
    def __init__(self):
        self.memory = deque(maxlen=config.BUFFER_SIZE)
        self.experience = recordtype('Experience', ["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self):
        experiences = random.sample(self.memory, k=config.BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
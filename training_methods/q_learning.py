import torch
from collections import deque
import copy
import numpy as np
import random

class QLearning:

    def __init__(self, policy_model, target_model, optimizer, 
                            criterion, buffer_size, batch_size):

        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.policy_model = policy_model
        self.target_model = target_model
        self.optimizer = optimizer
        self.criterion = criterion

    def sample_buffer(self):
        if len(self.buffer) <= self.batch_size:
            return list(self.buffer)
        else:
            return list(random.sample(self.buffer, self.batch_size))

    def learn(self, moves, result):

        for move in range(len(moves)-1):
            self.buffer.append((moves[move], (moves[move+1] + 0).detach()))
        
        self.buffer.append((moves[-1], torch.tensor(0 + result).reshape(-1)))

        train_sample = self.sample_buffer()
        
        target = [target for state, target in train_sample]
        state = [state for state, target in train_sample]

        self.optimizer.zero_grad()
        self.target_model.train()

        loss = self.criterion(torch.stack(target), torch.stack(state))

        loss.backward()
        self.optimizer.step()
    
    def update(model):
        pass
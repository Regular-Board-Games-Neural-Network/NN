import torch
from collections import deque
import copy
import numpy as np
import random

class QLearning:

    def __init__(self, model, optimizer, criterion):

        self.model = policy_model
        self.optimizer = optimizer
        self.criterion = criterion

    def learn(self, moves, result):

        transition = []
        for move in range(len(moves)-1):
            transition.append((moves[move], (moves[move+1] + 0).detach()))
        
        transition.append((moves[-1], torch.tensor(0 + result).reshape(-1)))
        
        target = [target for state, target in train_sample]
        state = [state for state, target in train_sample]

        self.optimizer.zero_grad()
        self.target_model.train()

        loss = self.criterion(torch.stack(target), torch.stack(state))

        loss.backward()
        self.optimizer.step()

        
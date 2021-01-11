import numpy as np

class MovePerformed:
    def __init__(self, state, action, next_state, action_state_value, reward = None):
        
        self.state = state
        self.action = action
        self.next_state = next_state
        self.action_state_value = action_state_value
        self.reward = reward

class ReplayMemory:
    
    def __init__(self, max_size = 1000):
        self.max_size = max_size
        self.memory = []
        self.position = 0

    def push(self, move):

        if self.position >= len(self.memory):
            self.memory.append(move)
        else:
            self.memory[self.position] = move

        self.position = (self.position + 1) % self.max_size
    
    def sample(self, sample_size):
        return np.random.choice(self.memory, sample_size)

    def last_id(self):
        return (self.position - 1 + self.max_size) % self.max_size

    def clear(self):
        self.memory = []
        self.position = 0
    
    def __getitem__(self, index):
        return self.memory[index]

    def __len__(self):
        return len(self.memory)


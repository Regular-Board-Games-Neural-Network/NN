import replay

class Training:
    def __init__(self, model, optimizer, criterion):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.memory = replay.ReplayMemory()
    
    def add_to_history(self, move):
        self.memory.push(move)
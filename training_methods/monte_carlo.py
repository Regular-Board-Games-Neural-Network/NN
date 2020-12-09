import torch

class MonteCarlo:

        def __init__(self, model, optimizer, criterion):

                self.model = model
                self.optimizer = optimizer
                self.criterion = criterion

        def learn(self, moves, result):

                self.model.train()
                self.model.zero_grad()

                self.loss = self.criterion(
                        torch.stack(moves).reshape(-1, 1), 
                        torch.full((len(moves), 1), result))

                self.loss.backward()
                self.optimizer.step()
import torch

class MonteCarlo:

        def __init__(self, model, optimizer, criterion):

                self.model = model
                self.optimizer = optimizer
                self.criterion = criterions

        def learn_monte_carlo(self, moves, result):

                self.model.train()
                self.model.zero_grad()

                self.loss = criterion(
                        torch.stack(moves).reshape(-1, 1), 
                        torch.full((len(moves), 1), result))

                self.loss.backward()
                self.optimizer.step()
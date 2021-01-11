import torch
import replay
from training_methods.training import Training

class MonteCarlo(Training):

        def __init__(self, model, optimizer, criterion):
                super(MonteCarlo, self).__init__(model, optimizer, criterion)

        def learn(self, result):

                self.model.train()
                self.model.zero_grad()

                move_values = [mv.action_state_value for mv in self.memory]
                self.memory.clear()

                self.loss = self.criterion(
                        torch.stack(move_values).reshape(-1, 1), 
                        torch.full((len(move_values), 1), result))

                self.loss.backward()
                self.optimizer.step()
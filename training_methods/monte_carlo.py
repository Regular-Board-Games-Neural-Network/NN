import torch
import replay
from training_methods.training import Training

class MonteCarlo(Training):

        def __init__(self, model, optimizer, criterion):
                super(MonteCarlo, self).__init__(model, optimizer, criterion)
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def learn(self, result):

                self.model.train()
                self.model.zero_grad()

                move_values = [mv.action_state_value for mv in self.memory]
                self.memory.clear()

                self.loss = self.criterion(
                        torch.stack(move_values).reshape(-1, 1).to(self.device), 
                        torch.full((len(move_values), 1), result).to(self.device))

                self.loss.backward()
                self.optimizer.step()
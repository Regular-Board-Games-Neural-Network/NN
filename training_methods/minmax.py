import torch
import rbg_game
import numpy as np
import model_utilis
from training_methods.training import Training

class MinMaxLearning(Training):

    def __init__(self, model, optimizer, criterion):
        super(MinMaxLearning, self).__init__(model, optimizer, criterion)
        
        self.device = torch.device('cpu')
        self.sample_size = 100

    def learn(self, result):
        
        if(len(self.memory) < self.sample_size):
            return
        
        action_sample = self.memory.sample(self.sample_size)
        
        target = []
        states = [model_utilis.parse_game_state(x.next_state) for x in action_sample]
        outs = self.model(torch.cat(states, 0).to(self.device))

        for x in action_sample:
            if x.reward is not None:
                target.append(torch.tensor(x.reward))
            else:
                rbs = rbg_game.resettable_bitarray_stack()
                moves = x.next_state.get_all_moves(rbs)
                boards = []
                
                for mv in moves:
                    boardcp = x.next_state.copy()
                    boardcp.apply_with_keeper(mv, rbs)
                    boards.append(model_utilis.parse_game_state(boardcp))

                values = self.model(torch.cat(boards, 0).to(self.device))

                
                target_value = torch.min(values).detach()
                target.append(target_value)

        self.model.zero_grad()
        self.model.train()

        loss = self.criterion(
            outs.reshape(-1, 1), 
            torch.stack(target).reshape(-1, 1))

        loss.backward()
        self.optimizer.step()

        
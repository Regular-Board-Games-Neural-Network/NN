import torch


def agent_learn(self, model, optimizer, criterion, moves, result):

        model.train()
        model.zero_grad()

        loss = criterion(torch.stack(self.moves_history), 
                                torch.full((len(self.moves_history), 1), result))
        
        loss.backward()
        optimizer.step()

        del moves
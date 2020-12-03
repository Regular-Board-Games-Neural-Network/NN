import torch


def learn_monte_carlo(model, optimizer, criterion, moves, result):

        model.train()
        model.zero_grad()

        loss = criterion(torch.stack(moves).reshape(-1, 1), torch.full((len(moves), 1), result))
        

        loss.backward()
        optimizer.step()

        del moves
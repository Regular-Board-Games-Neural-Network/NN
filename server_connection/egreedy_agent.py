import torch
import rbg
import utilities

class EgreedyAgent:

    def __init__(self, e_value):        
        
        self.device = torch.device('cpu')
        self.e_value = e_value

    def choose_action(self, state, model):
        
        moves = state.Moves()
        boards = []
        
        for mv in moves:
            revert = state.Apply(mv)
            boards.append(utilities.parse_game_state(state))
            state.Revert(revert)

        values = model(torch.cat(boards, 0).to(self.device))
        
        if torch.rand(1) > self.e_value:
            id = torch.argmax(values)
        else:
            id = torch.randperm(len(values))[:1][0]

        return moves[id], values[id]
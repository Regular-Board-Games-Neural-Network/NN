import torch
import rbg_game
import model_utilis

class EgreedyAgent:

    def __init__(self, e_value):        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.e_value = e_value

    def choose_action(self, state, model):
        
        rbs = rbg_game.resettable_bitarray_stack()
        moves = state.get_all_moves(rbs)
        boards = []
        
        for mv in moves:
            boardcp = state.copy()
            boardcp.apply_with_keeper(mv, rbs)
            boards.append(model_utilis.parse_game_state(boardcp))

        
        values = model(torch.cat(boards, 0).to(self.device))
        
        if torch.rand(1) > self.e_value:
            id = torch.argmax(values)
        else:
            id = torch.randperm(len(values))[:1][0]

        return moves[id], values[id]
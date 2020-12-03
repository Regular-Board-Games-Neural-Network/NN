import torch
import rbg_game
import model_utilis

class GreedyAgent:

    def __init__(self):        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def choose_action(self, state, model):
        
        rbs = rbg_game.resettable_bitarray_stack()
        moves = state.get_all_moves(rbs)
        boards = []
        
        for mv in moves:
            boardcp = state.copy()
            boardcp.apply_with_keeper(mv, rbs)
            boards.append(model_utilis.parse_game_state(boardcp))

        values = model(torch.cat(boards, 0).to(self.device))

        max_id = torch.argmax(values)

        return moves[max_id], values[max_id]
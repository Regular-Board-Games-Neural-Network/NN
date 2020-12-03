import torch
import rbg_game
import model_utilis
from model import ResModel
import torch.nn as nn

class Agent:

    def __init__(self, alpha, layer_size, num_of_layers, num_of_res_layers, number_of_filters):        
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.moves_history = []
        self.criterion = nn.MSELoss()

        self.model = ResModel(input_shape=layer_size, num_layers=num_of_layers, kernel_size=(3,3), 
            num_of_res_layers=num_of_res_layers, padding=(1, 1), 
            number_of_filters=number_of_filters).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

    def choose_action(self, state):
        
        rbs = rbg_game.resettable_bitarray_stack()
        moves = state.get_all_moves(rbs)
        boards = []
        
        for mv in moves:
            boardcp = state.copy()
            boardcp.apply_with_keeper(mv, rbs)
            boards.append(model_utilis.parse_game_state(boardcp))

        values = self.model(torch.cat(boards, 0).to(self.device))

        max_id = torch.argmax(values)
        self.moves_history.append(
            values[max_id]
        )

        return moves[max_id]

        
    def agent_learn(self, result):

        self.model.train()
        self.model.zero_grad()

        loss = self.criterion(torch.stack(self.moves_history).to(self.device), 
                                torch.full((len(self.moves_history), 1), result).to(self.device)).to(self.device)
        
        loss.backward()
        self.optimizer.step()

        self.moves_history = []
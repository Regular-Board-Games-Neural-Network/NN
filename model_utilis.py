import rbg_game
import torch
import numpy as np

def parse_game_state(state):

    def get_piece_val(cell):
        return state.get_piece(cell) / (pieces - 1)

    board_size = rbg_game.board_size()
    neighbours = rbg_game.board_degree()
    square_size = int(np.ceil(np.sqrt(neighbours + 1)))
    pieces = rbg_game.number_of_pieces()
    variables = rbg_game.number_of_variables()

    out = np.zeros((1, (board_size + variables), square_size, square_size))

    for i in range(board_size):
        submatrix = np.ones(square_size * square_size) * -1
        submatrix[0] = get_piece_val(i)
        
        for j in range(neighbours):
            neigh = rbg_game.get_neighbour(i, j)

            if neigh not in [-1, i]:
                submatrix[j + 1] = get_piece_val(neigh)

        submatrix = submatrix.reshape((square_size, square_size))
        out[:, i, :, :] = submatrix

    for i in range(variables):
        val = state.get_variable_value(i) / rbg_game.get_bound(i)
        submatrix = np.ones((square_size, square_size)) * val
        out[:, board_size + i, :, :] = submatrix

    return torch.tensor(out.astype(np.float32))

def save_model(path, model, layer_size, num_of_input_layers, num_of_res_layers, number_of_filters):
    torch.save(model.state_dict(), path + 'Resnet;{};{};{};{}'.format(
        layer_size, num_of_input_layers, num_of_res_layers, number_of_filters))

def load_model(path, model):
    model.load_state_dict(torch.load(path))
    return model
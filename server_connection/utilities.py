import rbg_game
import rbg
import torch
import numpy as np
import os

def get_input_shape():
    n = int(np.ceil(np.sqrt(rbg_game.board_degree() + 1))) 
    return (n, n)

def get_input_layers():
    return rbg_game.board_size() + rbg_game.number_of_variables()

def parse_game_state(state):

    board_content = state.board_content()

    def get_piece_val(cell):
        return board_content.at(cell) / (pieces - 1)

    board_size = rbg_game.board_size()
    neighbours = rbg_game.board_degree()
    square_size = int(np.ceil(np.sqrt(neighbours + 1)))
    pieces = rbg_game.number_of_pieces()
    variables = rbg_game.number_of_variables()

    parsed = np.zeros((1, (board_size + variables), square_size, square_size))

    for i in range(board_size):
        submatrix = np.ones(square_size * square_size) * -1
        submatrix[0] = get_piece_val(i)
        
        for j in range(neighbours):
            neigh = rbg_game.get_neighbour(i, j)

            if neigh not in [-1, i]:
                submatrix[j + 1] = get_piece_val(neigh)

        submatrix = submatrix.reshape((square_size, square_size))
        parsed[:, i, :, :] = submatrix

    variables_vals = state.variables_values()
    declarations = state.declarations()

    for i in range(variables):
        val = variables_vals[i] / declarations.variable_bound(i)
        submatrix = np.ones((square_size, square_size)) * val
        parsed[:, board_size + i, :, :] = submatrix

    return torch.tensor(parsed.astype(np.float32))

def save_model(path, model_name, model):
    torch.save(model.state_dict(), os.path.join(path, 'MODEL {}'.format(model_name)))

def load_model(path, model, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
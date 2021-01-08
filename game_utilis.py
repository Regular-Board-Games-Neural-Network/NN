import rbg_game
import random

def print_board(game_state):
    for x in range(8):
        print(' '.join([
            rbg_game.piece_to_string(game_state.get_piece(x * 8 + y)) for y in range(8)
        ]))

def print_vars(game_state):
    print(rbg_game.number_of_variables())
    vars = rbg_game.number_of_variables()
    
    for i in range(vars):
        print(rbg_game.variable_to_string(i), 
              game_state.get_variable_value(i), 
              rbg_game.get_bound(i))

def make_move(game_state, move, rbs):
    game_state.apply_with_keeper(move, rbs)
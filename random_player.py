import rbg_game
import random 
import utilis
from agent import Agent
from sys import stderr
import utilis
from tqdm import tqdm

rbs = rbg_game.resettable_bitarray_stack()
n = rbg_game.board_size()

def print_board(game_state):
    for x in range(8):
        print(' '.join([
            rbg_game.piece_to_string(game_state.get_piece(x * 8 + y)) for y in range(8)
        ]))

def print_vars(game_state):
    print(rbg_game.number_of_variables())
    vars = rbg_game.number_of_variables()
    for i in range(vars):
        print(rbg_game.variable_to_string(i), game_state.get_variable_value(i), rbg_game.get_bound(i))
    
def random_move(game_state):
    move = random.choice(game_state.get_all_moves(rbs))
    game_state.apply_with_keeper(move, rbs)

def make_move(game_state, move):
    game_state.apply_with_keeper(move, rbs)

num_games = 10000
nn_player_number = 2
nn_player = Agent(0.001)
wins = 0
draws = 0

for game_number in tqdm(range(num_games)):
    
    game_state = rbg_game.new_game_state()

    while True:
        # print_vars(game_state)
        # print_board(game_state)
        
        if game_state.get_current_player() == 0:
            break

        if game_state.get_current_player() == nn_player_number:
            move = nn_player.choose_action(game_state)
            make_move(game_state, move)
        else:
            random_move(game_state)
    
    nn_player.agent_learn(game_state.get_player_score(nn_player_number) / 100)

    if game_number >= num_games / 2:
        if game_state.get_player_score(nn_player_number) == 100:
            wins += 1
        if game_state.get_player_score(nn_player_number) == 50:
            draws += 1

print('Wins=', wins)
print('Draws=', draws)
print('Wins+Draws=', wins + draws)
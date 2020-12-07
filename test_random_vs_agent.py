import rbg_game
from model import ResModel
from tqdm import tqdm
import torch.nn as nn
from game_utilis import *
from model_utilis import *
from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from training_funcs import mt
import gc

rbs = rbg_game.resettable_bitarray_stack()
n = rbg_game.board_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha = 0.001
model = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)

model = load_model('/mnt/c/Users/zobni/Programming/NN/Resnet;(3, 3);66;2;256', model)

player_1 = GreedyAgent()
player_2 = RandomAgent()

num_games = 1
nn_player_number = 1
wins = 0
draws = 0

for game_number in tqdm(range(num_games)):
    
    game_state = rbg_game.new_game_state()
    moves_history = []

    while True:
        
        if game_state.get_current_player() == 0:
            break

        if game_state.get_current_player() == nn_player_number:
            move_rbs, move_value = player_1.choose_action(game_state, model)
            make_move(game_state, move_rbs, rbs)
            moves_history.append(move_value)
        else:
            move_rbs, move_value = player_2.choose_action(game_state)
            make_move(game_state, move_rbs, rbs)

        if game_state.get_player_score(nn_player_number) == 100:
            wins +=1
        if game_state.get_player_score(nn_player_number) == 50:
            wins +=1

    torch.cuda.empty_cache()
    gc.collect()

print('Wins=', wins)
print('Draws=', wins)
print('Wins+Draws=', wins)
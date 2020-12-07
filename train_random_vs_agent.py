import rbg_game
from model import ResModel
from tqdm import tqdm
import torch
import torch.nn as nn
from game_utilis import *
from model_utilis import *
from agents.egreedy_agent import EgreedyAgent
from agents.random_agent import RandomAgent
from training_methods.q_learning import QLearning

rbs = rbg_game.resettable_bitarray_stack()
n = rbg_game.board_size()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha = 0.001
policy_model = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)
target_model = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)

optimizer = torch.optim.Adam(target_model.parameters(), lr=model_alpha)
criterion = nn.MSELoss()

trainer = QLearning(policy_model, target_model, optimizer, 
                            criterion, 1000, 128)

player_1 = EgreedyAgent(e_value = 0.01)
player_2 = RandomAgent()

num_games = 1
nn_player_number = 1

for game_number in tqdm(range(num_games)):
    
    game_state = rbg_game.new_game_state()
    moves_history = []

    while True:
        
        if game_state.get_current_player() == 0:
            break

        if game_state.get_current_player() == nn_player_number:
            move_rbs, move_value = player_1.choose_action(game_state, policy_model)
            make_move(game_state, move_rbs, rbs)
            moves_history.append(move_value)
        else:
            move_rbs, move_value = player_2.choose_action(game_state)
            make_move(game_state, move_rbs, rbs)

    trainer.learn(moves_history, 1)

    exit(0)


    
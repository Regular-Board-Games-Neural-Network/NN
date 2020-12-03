import rbg_game
from model import ResModel
from tqdm import tqdm
from game_utilis import *
from model_utilis import *
from agents.egreedy_agent import EgreedyAgent
from agents.random_agent import RandomAgent


rbs = rbg_game.resettable_bitarray_stack()
n = rbg_game.board_size()

#nn_player = Agent(alpha = 0.001,layer_size=(3, 3), num_of_layers=(66), 
#                            num_of_res_layers=1, number_of_filters=256)

model = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=1, padding=(1, 1), 
            number_of_filters=256)

player_1 = EgreedyAgent(0.01)
player_2 = RandomAgent()

num_games = 10000
nn_player_number = 2

for game_number in tqdm(range(num_games)):
    
    game_state = rbg_game.new_game_state()

    while True:
        
        if game_state.get_current_player() == 0:
            break

        if game_state.get_current_player() == nn_player_number:
            move = nn_player.choose_action(game_state)
            make_move(game_state, move, rbs)
        else:
            random_move(game_state, rbs)
    
    
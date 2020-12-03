import rbg_game
from agent import Agent
from tqdm import tqdm
from game_utilis import *

rbs = rbg_game.resettable_bitarray_stack()
n = rbg_game.board_size()

num_games = 10000
nn_player_number = 2
nn_player = Agent(alpha = 0.001,layer_size=(3, 3), num_of_layers=(66), 
                            num_of_res_layers=1, number_of_filters=256)
wins = 0
draws = 0

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
    
    nn_player.agent_learn(game_state.get_player_score(nn_player_number) / 100)

    if game_number >= num_games / 2:
        if game_state.get_player_score(nn_player_number) == 100:
            wins += 1
        if game_state.get_player_score(nn_player_number) == 50:
            draws += 1

print('Wins=', wins)
print('Draws=', draws)
print('Wins+Draws=', wins + draws)
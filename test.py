import rbg_game
from tqdm import tqdm
from agents.random_agent import RandomAgent
from game_utilis import *
from model_utilis import *

def test(test_config):

    rbs = rbg_game.resettable_bitarray_stack()

    model_1 = test_config['model_1']
    player_1 = test_config['player_1']

    model_2 = test_config['model_2']
    player_2 = test_config['player_2']

    num_games = test_config['num_games']

    player_1_number = 1
    player_2_number = 2

    wins = 0
    draws = 0

    for game_number in tqdm(range(num_games)):
        
        game_state = rbg_game.new_game_state()

        while True:
            
            if game_state.get_current_player() == 0:
                break

            if game_state.get_current_player() == player_1_number:
                move_rbs, _ = player_1.choose_action(game_state, model_1)
                make_move(game_state, move_rbs, rbs)
            else:
                move_rbs, _ = player_2.choose_action(game_state, model_2)
                make_move(game_state, move_rbs, rbs)

        if game_state.get_player_score(player_1_number) == 100:
            wins +=1
        if game_state.get_player_score(player_1_number) == 50:
            draws +=1

        player_2_number, player_1_number = player_1_number, player_2_number

    print('Wins=', wins)
    print('Draws=', draws)
    print('Wins+Draws=', wins+draws)
    print('Win+Draw rate=', (wins+draws) / num_games)
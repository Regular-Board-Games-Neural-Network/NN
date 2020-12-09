import rbg_game
from tqdm import tqdm
from agents.random_agent import RandomAgent
from game_utilis import *

def train_vs_random(train_config):

    rbs = rbg_game.resettable_bitarray_stack()

    model = train_config['model']
    trainer = train_config['trainer']
    player_1 = train_config['player']

    player_2 = RandomAgent()

    num_games  = train_config['num_games']
    save_model_every_n_iterations = train_config['save_model_every_n_iterations']

    player_1_number = 1
    player_2_number = 2
    
    for game_number in tqdm(range(num_games)):
    
        game_state = rbg_game.new_game_state()
        moves_history = []

        while True:
            
            if game_state.get_current_player() == 0:
                break

            if game_state.get_current_player() == player_1_number:
                move_rbs, move_value = player_1.choose_action(game_state, model)
                make_move(game_state, move_rbs, rbs)
                moves_history.append(move_value)
            else:
                move_rbs, move_value = player_2.choose_action(game_state)
                make_move(game_state, move_rbs, rbs)
        
        trainer.learn(moves_history, game_state.get_player_score( player_1_number)/100)

        player_2_number, player_1_number = player_1_number, player_2_number
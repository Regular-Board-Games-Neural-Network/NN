import replay
import rbg_game
from tqdm import tqdm
from agents.random_agent import RandomAgent
from game_utilis import *
from model_utilis import *

def train(train_config):

    rbs = rbg_game.resettable_bitarray_stack()

    model_1 = train_config['model_1']
    trainer_1 = train_config['trainer_1']
    player_1 = train_config['player_1']

    model_2 = train_config['model_2']
    trainer_2 = train_config['trainer_2']
    player_2 = train_config['player_2']

    model_name = train_config['model_name']
    num_games = train_config['num_games']
    save_model_every_n_iterations = train_config['save_model_every_n_iterations']
    save_path = train_config['save_path']

    player_1_number = 1
    player_2_number = 2
    
    for game_number in tqdm(range(num_games)):
    
        game_state = rbg_game.new_game_state()

        while True:
            
            if game_state.get_current_player() == 0:
                break

            old_state = game_state.copy()

            if game_state.get_current_player() == player_1_number:
                move_rbs, move_value = player_1.choose_action(game_state, model_1)
                make_move(game_state, move_rbs, rbs)
                
                if game_state.get_current_player() == 0:
                    trainer_1.add_to_history(
                        replay.MovePerformed(
                            old_state, 
                            move_rbs, 
                            game_state.copy(), 
                            move_value,
                            game_state.get_player_score(player_1_number)/100))
                else:
                    trainer_1.add_to_history(
                        replay.MovePerformed(
                            old_state, 
                            move_rbs, 
                            game_state.copy(), 
                            move_value))
            else:
                move_rbs, move_value = player_2.choose_action(game_state, model_2)
                make_move(game_state, move_rbs, rbs)
                
                if game_state.get_current_player() == 0:
                    trainer_2.add_to_history(
                        replay.MovePerformed(
                            old_state, 
                            move_rbs, 
                            game_state.copy(), 
                            move_value,
                            game_state.get_player_score(player_2_number)/100))
                else:
                    trainer_2.add_to_history(
                        replay.MovePerformed(
                            old_state, 
                            move_rbs, 
                            game_state.copy(), 
                            move_value))
        
        trainer_1.learn(game_state.get_player_score(player_1_number)/100)
        trainer_2.learn(game_state.get_player_score(player_2_number)/100)

        player_2_number, player_1_number = player_1_number, player_2_number

        if game_number % save_model_every_n_iterations== 0:
            save_model(save_path, model_name+'_model_1', model_1)
            save_model(save_path, model_name+'_model_2', model_2)

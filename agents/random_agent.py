import rbg_game
import random

class RandomAgent:        

    def choose_action(self, game_state):
        
        rbs = rbg_game.resettable_bitarray_stack()
        move = random.choice(game_state.get_all_moves(rbs))
        
        return (move, None)
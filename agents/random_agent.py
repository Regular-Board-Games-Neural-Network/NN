import rbg_game

class RandomAgent:        

    def choose_action(self, state, model):
        
        rbs = rbg_game.resettable_bitarray_stack()
        move = random.choice(game_state.get_all_moves(rbs))
        
        return (move, None)
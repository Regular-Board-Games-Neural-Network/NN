from rbg import *
from train import train
from models.resnet import ResModel
from egreedy_agent import EgreedyAgent
from utilities import *
import torch.nn as nn
import argparse
import sys
import torch
 
def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("host",help="The host to connect to.")
    parser.add_argument("port",help="The port to connect to.")
    args = parser.parse_args()

    client = Client(args.host, args.port)
    state = CreateGameState(client.description())
    client.Ready()

    player = client.player()
    print('I am player',state.declarations().players_resolver().Name(player))
    
    model = ResModel(input_shape=get_input_shape(),  num_layers=get_input_layers(), 
                num_of_res_layers=int(2), number_of_filters=int(256)).to(device)
    
    player_policy = EgreedyAgent(e_value=0.02)

    #load model
    load_model('data/model/RANDOM_(3, 3)_66_2_256.zip', model, device)
    
    moves = state.Moves()

    while moves:
        if state.current_player() == player:
            client.ReadDeadline()
            move, _ = player_policy.choose_action(state, model)
            state.Apply(move)
            client.Write(move)
        else:
            move = client.Read()
            state.Apply(move)
        
        moves = state.Moves()

if __name__=="__main__":
    main()

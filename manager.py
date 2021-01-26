import argparse
import torch
import torch.nn as nn
from train import train
from test impor test
from models.resnet import ResModel
from agents.egreedy_agent import EgreedyAgent
from agents.greedy_agent import GreedyAgent
from training_methods.monte_carlo import MonteCarlo
from model_utilis import *

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('model_alpha', metavar=1, type=float, required=True,
                    help='model_alpha')
parser.add_argument('num_of_res_layers', metavar=1, type=int, required=True,
                    help='num_of_res_layers')                    
parser.add_argument('number_of_filters', metavar=1, type=int, required=True,
                    help='number_of_filters')
parser.add_argument('agent', metavar=1, required=True,
                    help='agent of choice')
parser.add_argument('num_games', metavar=1, type=int, required=True,
                    help='num_games')
parser.add_argument('save_model_every_n_iterations', metavar=1, type=int, required=True,
                    help='save_model_every_n_iterations')
parser.add_argument('model_name', metavar=1, required=True,
                    help='model_name')
parser.add_argument('save_path', metavar=1, required=True,
                    help='save_path')
parser.add_argument('-opt', type=float, metavar=1,
                    help='Argument for agent, to activate write: -opt arg')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha_1 = args.model_alpha[0]
model_1 = ResModel(input_shape=get_input_shape(),  num_layers=get_input_layers(), 
            num_of_res_layers=args.num_of_res_layers[0], number_of_filters=args.number_of_filters[0]).to(device)

optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=model_alpha_1)

model_alpha_2 = args.model_alpha[0]
model_2 = ResModel(input_shape=get_input_shape(), num_layers=get_input_layers(), 
            num_of_res_layers=args.num_of_res_layers[0], number_of_filters=args.number_of_filters[0]).to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=model_alpha_2)

criterion = nn.MSELoss()

trainer_1 = MonteCarlo(model_1, optimizer_1, criterion)
trainer_2 = MonteCarlo(model_2, optimizer_2, criterion)

if args.agent[0] == "EgreedyAgent":
	player_1 = EgreedyAgent(e_value = args.opt)
	player_2 = EgreedyAgent(e_value = args.opt)
elif args.agent[0] == "GreedyAgent":
	player_1 = GreedyAgent()
	player_2 = GreedyAgent()
elif args.agent[0] == "RandomAgent":
	player_1 = RandomAgent() 
	player_2 = RandomAgent()
else:
	print("Wrong agent")
	exit()

train({'trainer_1': trainer_1, 'model_1': model_1, 'player_1': player_1,
        'trainer_2': trainer_2, 'model_2': model_2, 'player_2': player_2,
        'num_games': args.num_games[0], 'save_model_every_n_iterations': args.save_model_every_n_iterations[0], 'model_name': args.model_name[0],
        'save_path': args.save_path[0]})
'''
Kolejność argumentów:
model_alpha, num_of_res_layers, number_of_filters, agent, agent_ option, num_games, save_model_every_n_iterations, model_name, save_path

model_1 = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)

model_1 = load_model('/mnt/c/Users/zobni/Programming/NN/data/model/NN1VSNN2_(3, 3)_66_2_256.zip', model_1, device)

model_2 = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)

model_2 = load_model('/mnt/c/Users/zobni/Programming/NN/data/model/NN1VSNN1_(3, 3)_66_2_256.zip', model_2, device)

player_1 = GreedyAgent()
player_2 = GreedyAgent()

test({ 'model_1': model_1, 'player_1': player_1, 'model_2': model_2, 'player_2': player_2,
       'num_games': 30})
'''

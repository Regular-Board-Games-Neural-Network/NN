import sys
import torch
import torch.nn as nn
from train import train
from models.resnet import ResModel
from agents.egreedy_agent import EgreedyAgent
from agents.greedy_agent import GreedyAgent
from training_methods.monte_carlo import MonteCarlo
from training_methods.q_learning import QLearning
from model_utilis import *

n = len(sys.argv)
if n != 12 and n != 11:
	print("error: wrong number of arguments :(")
	exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha_1 = int(sys.argv[1])
model_1 = ResModel(input_shape=tuple(sys.argv[2]), num_layers=int(sys.argv[3]), 
            num_of_res_layers=int(sys.argv[4]), number_of_filters=int(sys.argv[5])).to(device)

optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=model_alpha_1)

model_alpha_2 = int(sys.argv[1])
model_2 = ResModel(input_shape=tuple(sys.argv[2]), num_layers=int(sys.argv[3]), 
            num_of_res_layers=int(sys.argv[4]), number_of_filters=int(sys.argv[5])).to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=model_alpha_2)

criterion = nn.MSELoss()

trainer_1 = QLearning(model_1, optimizer_1, criterion)
trainer_2 = QLearning(model_2, optimizer_2, criterion)

if sys.argv[6] == "EgreedyAgent":
	player_1 = EgreedyAgent(e_value = int(sys.argv[7]))
	player_2 = EgreedyAgent(e_value = int(sys.argv[7]))
elif sys.argv[6] == "GreedyAgent":
	player_1 = GreedyAgent()
	player_2 = GreedyAgent()
elif sys.argv[6] == "RandomAgent":
	player_1 = RandomAgent()
	player_2 = RandomAgent()
else:
	print("Wrong agent")
	exit()

train({'trainer_1': trainer_1, 'model_1': model_1, 'player_1': player_1,
        'trainer_2': trainer_2, 'model_2': model_2, 'player_2': player_2,
        'num_games': sys.argv[n-4], 'save_model_every_n_iterations': sys.argv[n-3], 'game_name': sys.argv[n-2],
        'save_path': sys.argv[n-1]})
'''
Kolejność argumentów:
model_alpha, input_shape, num_layers, num_of_res_layers, number_of_filters, agent, agent_ option, num_games, savae_model_every_n_iterations, game_name, save_path

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

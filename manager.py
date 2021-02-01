import argparse
import torch
import torch.nn as nn
from train import train
from test import test
from models.resnet import ResModel
from agents.egreedy_agent import EgreedyAgent
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from training_methods.monte_carlo import MonteCarlo
from model_utilis import *

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--mode', metavar=1,
                    required=True, help='Train/Test', choices=['Train', 'Test'])
parser.add_argument('--model_alpha', metavar=1, type=float,
                    required=True, help='model_alpha')
parser.add_argument('--num_of_res_layers', metavar=1, type=int, required=True,
                    help='num_of_res_layers')
parser.add_argument('--number_of_filters', metavar=1, type=int, required=True,
                    help='number_of_filters')
parser.add_argument('--agent_1', metavar=1, required=True,
                    help='agent of choice')
parser.add_argument('--agent_2', metavar=1, required=True,
                    help='agent of choice')
parser.add_argument('--num_games', metavar=1, type=int, required=True,
                    help='num_games')
parser.add_argument('--save_model_every_n_iterations', metavar=1, type=int, required=True,
                    help='save_model_every_n_iterations')
parser.add_argument('--model_name', metavar=1, required=True,
                    help='model_name')
parser.add_argument('--save_path', metavar=1, required=True,
                    help='save_path')
parser.add_argument('-opt', type=float, metavar=1,
                    help='Argument for agent, to activate write: -opt arg')
parser.add_argument('-load_1', metavar=1,
                    help='Load model 1', default=None)
parser.add_argument('-load_2', metavar=1,
                    help='Load model 2', default=None)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha_1 = args.model_alpha
model_1 = ResModel(input_shape=get_input_shape(),  num_layers=get_input_layers(),
                   num_of_res_layers=args.num_of_res_layers, number_of_filters=args.number_of_filters).to(device)

optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=model_alpha_1)

model_alpha_2 = args.model_alpha
model_2 = ResModel(input_shape=get_input_shape(), num_layers=get_input_layers(),
                   num_of_res_layers=args.num_of_res_layers, number_of_filters=args.number_of_filters).to(device)

optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=model_alpha_2)

criterion = nn.MSELoss()

trainer_1 = MonteCarlo(model_1, optimizer_1, criterion)
trainer_2 = MonteCarlo(model_2, optimizer_2, criterion)

if args.agent_1 == "EgreedyAgent":
    player_1 = EgreedyAgent(e_value=args.opt)
elif args.agent_1 == "GreedyAgent":
    player_1 = GreedyAgent()
elif args.agent_1 == "RandomAgent":
    player_1 = RandomAgent()
else:
    print("wrong agent")

if args.agent_2 == "EgreedyAgent":
    player_2 = EgreedyAgent(e_value=args.opt)
elif args.agent_2 == "GreedyAgent":
    player_2 = GreedyAgent()
elif args.agent_2 == "RandomAgent":
    player_2 = RandomAgent()
else:
    print("wrong agent")

if args.load_1:
    model_1 = load_model(args.load_1, model_1, device)

if args.load_2:
    model_2 = load_model(args.load_2, model_2, device)


if args.mode == 'Train':
    train({'trainer_1': trainer_1, 'model_1': model_1, 'player_1': player_1,
           'trainer_2': trainer_2, 'model_2': model_2, 'player_2': player_2,
           'num_games': args.num_games, 'save_model_every_n_iterations': args.save_model_every_n_iterations, 'model_name': args.model_name,
           'save_path': args.save_path})

else:
    test({'model_1': model_1, 'player_1': player_1, 'model_2': model_2, 'player_2': player_2,
          'num_games': args.num_games})

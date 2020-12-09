from train import train_vs_random
import torch
import torch.nn as nn
from model import ResModel
from agents.egreedy_agent import EgreedyAgent
from training_methods.monte_carlo import MonteCarlo


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_alpha = 0.001
model = ResModel(input_shape=(3, 3), num_layers=66, kernel_size=(3,3), 
            num_of_res_layers=2, padding=(1, 1), 
            number_of_filters=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=model_alpha)
criterion = nn.MSELoss()
trainer = MonteCarlo(model, optimizer, criterion)

player_1 = EgreedyAgent(e_value = 0.01)

train_vs_random({'trainer': trainer, 'model': model, 'player': player_1,
                 'num_games': 30, 'save_model_every_n_iterations': 100})

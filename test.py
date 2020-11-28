import torch
from model import Model

test = Model(3)

test_input = [0.5,0.5,0.5]

print(test.forward(test_input))
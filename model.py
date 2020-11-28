import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_size):

        super(Model, self).__init__()
        self.MLP = nn.Sequential()
        self.MLP.add_module('layer1', nn.Linear(input_size, 128))
        self.MLP.add_module('relu1', nn.ReLU(True))
        self.MLP.add_module('layer2', nn.Linear(128, 256))
        self.MLP.add_module('relu2', nn.ReLU(True))
        self.MLP.add_module('layer3', nn.Linear(256, 256))
        self.MLP.add_module('relu3', nn.ReLU(True))
        self.MLP.add_module('output', nn.Linear(256, 1))

    def forward(self, input):
        
        return self.MLP(input)
        


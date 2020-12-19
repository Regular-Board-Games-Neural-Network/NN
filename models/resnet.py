import torch.nn as nn
import torch
import numpy as np
import random 


class Res_block(nn.Module):

    def __init__(self, inplanes, kernel_size, padding):
        super(Res_block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes,
                               kernel_size, padding=padding)
        self.batchn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, inplanes,
                               kernel_size, padding=padding)
        self.batchn2 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_x):

        identity = input_x

        out = self.conv1(input_x)
        out = self.batchn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchn2(out)

        
        out += identity
        out = self.relu(out)

        return out

class ResModel(nn.Module):

    def __init__(self, input_shape, num_layers, num_of_res_layers, number_of_filters):
        super(ResModel, self).__init__()

        kernel_size = (3, 3)
        padding = (1, 1)

        self.model = nn.Sequential()
        self.model.add_module('Conv Start', 
                    nn.Conv2d(num_layers, number_of_filters, kernel_size=kernel_size, padding=padding))
        
        self.model.add_module('Batch_norm', nn.BatchNorm2d(number_of_filters))
        self.model.add_module('relu', nn.ReLU(inplace=True))

        for i in range(0,num_of_res_layers):
            self.model.add_module('Res_Layer {}'.format(i), 
                    Res_block(number_of_filters, kernel_size, padding))

        self.model.add_module('Conv Flat', nn.Conv2d(number_of_filters, 8, kernel_size=(1, 1)))
        self.model.add_module('Flatten', nn.Flatten())
        self.model.add_module('Output', nn.Linear(8 * input_shape[0] * input_shape[1], 1))


    def forward(self, input_x):

        return self.model(input_x)



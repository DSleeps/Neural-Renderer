import torch
from torch import nn

class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, position_dim=3):
        super(Decoder, self).__init__()

        self.ff = nn.Sequential(nn.Linear(input_size + position_dim,hidden_size))
        for i in range(num_layers-1):
            self.ff.add_module('layer_' + str(i), nn.Linear(hidden_size, hidden_size))
            self.ff.add_module('ReLU_' + str(i), nn.ReLU())
        self.ff.add_module('layer_' + str(i), nn.Linear(hidden_size, output_size))
        self.ff.add_module('ReLU_' + str(i), nn.ReLU())

    def forward(self, inputs, desired_positions):
        return self.ff(torch.cat((inputs, desired_positions), dim=1))

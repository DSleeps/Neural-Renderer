import torch
from torch import nn
'''
class Decoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, position_dim=3):
        super(Decoder, self).__init__()

        self.ff = nn.Sequential(nn.Linear(input_size + position_dim,hidden_size))
        for i in range(num_layers-1):
            self.ff.add_module('layer_' + str(i), nn.Linear(hidden_size, hidden_size))
            self.ff.add_module('ReLU_' + str(i), nn.ReLU())
        self.ff.add_module('layer_' + str(i), nn.Linear(hidden_size, output_size))
        self.ff.add_module('ReLU_' + str(i), nn.Sigmoid())

    def forward(self, inputs, desired_positions):
        inputs = inputs * 0 # THIS IS TEMPORARY TO TEST THIS THING
        return self.ff(torch.cat((inputs, desired_positions), dim=1))

'''
class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size, position_dim=3):
        super(Decoder, self).__init__()
        self.dense = nn.Sequential(nn.Linear(input_size + position_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, 1024),
                                   nn.ReLU())

        self.reverse_CNN = nn.Sequential(nn.ConvTranspose2d(4, 6, kernel_size=3, stride=1, padding=1),
                                         nn.ConvTranspose2d(6,5, kernel_size=4, stride=2, padding=1),
                                         nn.ConvTranspose2d(5,3, kernel_size=4, stride=2, padding=1))

        # self.final = nn.Sequential(nn.Linear(output_size, output_size), nn.Sigmoid())

    def forward(self, inputs, desired_positions):
        batch_size = inputs.shape[0]
        
        output = self.dense(torch.cat((inputs, desired_positions), dim=1))
        output = torch.reshape(output, (batch_size, 4, 16, 16))

        output = self.reverse_CNN(output)
        
        output = torch.reshape(output, (batch_size, output.shape[1]*output.shape[2]*output.shape[3]))
        # output = self.final(output)

        return nn.Sigmoid()(output)

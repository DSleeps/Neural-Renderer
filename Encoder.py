import torch
from torch import nn

class Encoder(nn.Module):
    
    def __init_(self, input_size, hidden_size, output_size):
        # Idk why this is here actually, maybe to do the nn.Module init
        super(Encoder, self).__init__()

        # This is the CNN that will encode each image
        self.CNN_1 = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv3d(3, 3, kernel_size=3, stride=1))

        self.CNN_2 = nn.Sequential(

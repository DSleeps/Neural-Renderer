import torch
from torch import nn

class Encoder(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_size, device):
        # Idk why this is here actually, maybe to do the nn.Module init
        super(Encoder, self).__init__()
        
        self.device = device
        self.position_dim = 3
        self.output_size = output_size

        # This is the CNN that will encode each image
        self.strides_1 = [2, 2]
        self.channel_out_1 = 128
        self.CNN_1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, stride=self.strides_1[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(256, self.channel_out_1, kernel_size=3, stride=self.strides_1[1], padding=1),
            nn.ReLU())
        
        # This CNN will take in the positional arguments as well as the output
        # of CNN_1
        self.strides_2 = [1,2]
        self.channel_out_2 = 128
        self.pool_kernel_2 = 16
        self.CNN_2 = nn.Sequential(
            nn.Conv2d(self.channel_out_1 + self.position_dim, 128, kernel_size=3, stride=self.strides_2[0], padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.channel_out_2, kernel_size=4, stride=self.strides_2[1], padding=1),
            nn.ReLU())
        
        # Calculate the dimensions of the image after going through CNN_1
        self.initial = input_size
        for stride in self.strides_1:
            self.initial = int(self.initial/stride)
        
        # Calculate the output length of the CNN_2
        self.output_length = self.initial
        for stride in self.strides_2:
            self.output_length = int(self.output_length/stride)
        self.output_length = self.output_length**2 * self.channel_out_2

        # The fully connected layer after the CNN
        self.fully_connected = nn.Linear(self.output_length, output_size)

        # This is the LSTM that takes the output of the second CNN_2 and encodes
        # it into a vector
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1, bidirectional=False)
    
    # Where inputs are the input images, im_positions are the positions of the camera
    # at the given images, image_num is the number of images for each scene, and
    # desired positon is a list of the desired positions to be tested
    def forward(self, inputs, im_positions):
        batch_size = inputs.shape[0]
        
        # Calculate the dimensions of the image after going through CNN_1
        initial = inputs.shape[3]
        for stride in self.strides_1:
            initial = int(initial/stride)
        
        # Calculate the output length of the CNN_2
        output_length = self.channel_out_2 * initial
        for stride in self.strides_2:
            output_length = int(output_length/stride)

        CNN_outs = torch.zeros(inputs.shape[1], batch_size, self.output_size).to(self.device)
        for i in range(inputs.shape[1]):
            output = self.CNN_1(inputs[:,i,:,:])
            
            im_pos = torch.reshape(im_positions[:,i,:], (batch_size,1,1,im_positions.shape[2]))
            concat = torch.zeros(batch_size, self.position_dim, self.initial, self.initial).to(self.device)
            concat[:] = im_pos.permute(0,3,1,2)

            output = torch.cat((output, concat), dim=1)
            output = self.CNN_2(output)
            
            # Reshape the output and feed it through a fully connected layer
            output = torch.reshape(output, (batch_size, self.output_length))
            output = self.fully_connected(output)
            CNN_outs[i] = output

        output, hiddens = self.lstm(CNN_outs)
        output = output[-1]
        return output


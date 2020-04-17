import torch
from torch import nn
import numpy as np

from Encoder import Encoder
from Decoder import Decoder
from LoadDataset import load_data, sample_data

input_size = 64
encoding_size = 512
hidden_size = 512

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(input_size, encoding_size, hidden_size)
        self.decoder = Decoder(hidden_size, 512, 2, input_size**2 * 3)

    def forward(self, inputs, im_positions, desired_positions):
        return self.decoder(self.encoder(inputs, im_positions), desired_positions)


# For testing purposes
'''
inputs = torch.randn(batch_size, max_image_num, 3, 64, 64)
im_positions = torch.randn(batch_size, max_image_num, position_dim)
desired_positions = torch.randn(batch_size, position_dim)
print(inputs.dtype)
print(im_positions.dtype)
print(desired_positions.dtype)

encoder = Encoder(64, encoding_size, hidden_size)
decoder = Decoder(hidden_size, 512, 2, input_size**2 * 3)
print(decoder(encoder(inputs, im_positions), desired_positions))
'''

# Initialize mode
model = EncoderDecoder()

# Training parameters
batch_size = 25
max_image_num = 10
iteration_count = 10000
learning_rate = 0.0001

# Load the dataset
images, positions = load_data()

# Initialize the loss and the optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(iteration_count):
    b_images, b_im_positions, b_test_images, b_test_im_positions = sample_data(batch_size, np.random.randint(1,max_image_num), images, positions)

    outputs = model(b_images, b_im_positions, b_test_im_positions)
    desired = torch.reshape(b_test_images, (batch_size, input_size**2 * 3))

    loss = loss_fn(outputs, desired)
    
    if (i % 100 == 0):
        print('Iteration ' + str(i))
        print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


import torch
from torch import nn
import numpy as np
from PIL import Image

from Encoder import Encoder
from Decoder import Decoder
from LoadDataset import load_data, sample_data

input_size = 64
encoding_size = 512
hidden_size = 512

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(input_size, encoding_size, hidden_size, device)
        self.decoder = Decoder(hidden_size, 512, 2, input_size**2 * 3)

    def forward(self, inputs, im_positions, desired_positions):
        return self.decoder(self.encoder(inputs, im_positions), desired_positions)

'''
# For testing purposes

inputs = torch.randn(batch_size, max_image_num, 3, 64, 64).to(device)
im_positions = torch.randn(batch_size, max_image_num, position_dim).to(device)
desired_positions = torch.randn(batch_size, position_dim).to(device)

encoder = Encoder(64, encoding_size, hidden_size, device).to(device)
decoder = Decoder(hidden_size, 512, 2, input_size**2 * 3).to(device)
print(decoder(encoder(inputs, im_positions), desired_positions))
'''
if __name__ == '__main__':
    # Initialize mode
    model = EncoderDecoder().to(device)

    # Training parameters
    batch_size = 40
    max_image_num = 10
    iteration_count = 15000
    learning_rate = 0.0005

    # Load the dataset
    print('Loading dataset...')
    images, positions = load_data(device)
    print('Loaded!')

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
            
            # model.encoder.prints = True
            outputs = model(b_images[:1], b_im_positions[:1], b_test_im_positions[:1])
            # model.encoder.prints = False
            if (i % 1000 == 0):
                sample_im = np.reshape(outputs[0].cpu().data.numpy() * 255., (3, input_size, input_size)).astype(np.uint8)
                desired_im = np.reshape(desired[0].cpu().data.numpy() * 255., (3, input_size, input_size)).astype(np.uint8)
                
                im = Image.fromarray(np.transpose(sample_im, (1,2,0))).save('sample_' + str(i) + '.jpg')
                Image.fromarray(np.transpose(desired_im, (1,2,0))).save('desired_' + str(i) + '.jpg')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save the resulting model
    torch.save(model.state_dict(), 'FirstModel.t')

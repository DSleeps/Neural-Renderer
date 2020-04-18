import torch
from torch import nn
import numpy as np
import pygame

from Encoder import Encoder
from Decoder import Decoder
from Train import EncoderDecoder
from LoadDataset import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"

images, positions = load_data(device)

input_size = 64
encoding_size = 512
hidden_size = 512

model = EncoderDecoder().to(device)
model.load_state_dict(torch.load('FirstModel.t'))
model.eval()

# Now init the screen and stuffz
pygame.init()

im_size = 512
display = pygame.display.set_mode((im_size, im_size))

# The variables for the position and the angle
x = 0
z = 0
a = 0

move = [0,0]
move_speed = 0.1

turn = 0
turn_speed = 0.1
cur_pos = torch.zeros(1,3).to(device)

inputs = images[:1,:5,:,:,:]
in_pos = positions[:1,:5,:]
while (True):
    # Get player key presses
    for event in pygame.event.get():
        if (event.type == pygame.KEYDOWN):
            if (event.key == pygame.K_w):
                move[0] = -1
            elif (event.key == pygame.K_s):
                move[0] = 1
            elif (event.key == pygame.K_a):
                move[1] = 1
            elif (event.key == pygame.K_d):
                move[1] = -1
            elif (event.key == pygame.K_j):
                print('press')
                turn = 1
            elif (event.key == pygame.K_l):
                turn = -1
        elif (event.type == pygame.KEYUP):
            if (event.key == pygame.K_w):
                move[0] = 0
            elif (event.key == pygame.K_s):
                move[0] = 0
            elif (event.key == pygame.K_a):
                move[1] = 0
            elif (event.key == pygame.K_d):
                move[1] = 0
            elif (event.key == pygame.K_j):
                turn = 0
                print('lift')
            elif (event.key == pygame.K_l):
                turn = 0

    x += move[0] * move_speed * np.cos(a) + move[1] * move_speed * np.cos(a + np.pi/2)
    z += move[0] * move_speed * np.sin(a) + move[1] * move_speed * np.sin(a + np.pi/2)
    a += turn * turn_speed
    a = a % (2*np.pi)
    
    print('Cur pos')
    print(x)
    print(z)
    print(a)
    
    # Generate the image
    cur_pos[0][0] = x
    cur_pos[0][1] = z
    cur_pos[0][2] = a
    output = model(inputs, in_pos, cur_pos)
    sample_im = np.reshape(output[0].cpu().data.numpy() * 255., (3, input_size, input_size)).astype(np.uint8)
    sample_im = np.transpose(sample_im, (1,2,0))

    # Draw the image
    ratio = im_size/input_size
    for im_x in range(input_size):
        for im_y in range(input_size):
            p = sample_im[im_y][im_x]
            pygame.draw.rect(display, (p[0], p[1], p[2]), (im_x*ratio, im_y*ratio, ratio, ratio))

    pygame.display.update()


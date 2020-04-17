import torch
import pickle
import numpy as np
from PIL import Image

dataset_folder = 'ThreeDDataset/'
image_width = 64
images_per_scene = 20

def load_data(device):
    # Load the positions and convert them to a tensor
    positions = None
    with open('positions.pickle', 'rb') as f:
        positions = pickle.load(f)
        positions = torch.from_numpy(np.float32(np.asarray(positions))).to(device)
    
    # Load the images and convert them all to tensors
    images = torch.zeros(positions.shape[0], images_per_scene, 3, image_width, image_width).to(device)
    for i in range(positions.shape[0]):
        # This is the number of images per scene
        for j in range(images_per_scene):
            im = Image.open(dataset_folder + 'scene_' + str(i) + '_' + str(j) + '.png')
            im = torch.from_numpy(np.array(im.convert('RGB')))
            images[i][j] = im.permute(2,0,1)
    
    return images, positions
           
def sample_data(batch_size, num_images, dataset, positions):
    scene_indices = np.random.choice(dataset.shape[0], batch_size)
    image_choices = np.random.choice(dataset.shape[1], num_images + 1)
     
    images = dataset[scene_indices][:,image_choices[:-1],:,:]
    test_images = dataset[scene_indices][:,image_choices[-1],:,:]
    
    im_positions = positions[scene_indices][:,image_choices[:-1],:]
    test_im_positions = positions[scene_indices][:,image_choices[-1],:]

    return images, im_positions, test_images, test_im_positions

import vpython as vp
import numpy as np
import random as r
import time
import pickle

# The number of scenes to generate
data_points = 1000 

# Number of samples per scene
samples_per_scene = 20

# The number to start on
start_num = 0

# The width and the height of the images generated
width = 64
height = 64

# The maximum number of objects in the room
max_obs = 3

# The room parameters
room_width = 40
thickness = 0.2
wall_height = 10

cam_height = 6

# The fraction away from the wall that an object should be
room_frac = 0.75

color_options = [vp.color.green, vp.color.yellow, vp.color.magenta, vp.color.orange, vp.color.purple, vp.color.white]

# The object parameters
max_height = 8
min_height = 4
max_width = 6
min_width = 3

# Generate the objects but make them invisible for now
all_objects = []
for i in range(max_obs):
    all_objects.append(vp.box(visible=False))
for i in range(max_obs):
    all_objects.append(vp.cone(visible=False))
for i in range(max_obs):
    all_objects.append(vp.sphere(visible=False))

# Set the width and height of the window
scene = vp.scene
scene.width = width
scene.height = height

# Set the color of the scene
scene.background = vp.color.cyan

# First draw the room
floor = vp.box(pos=vp.vector(0,0,0), size=vp.vector(room_width,thickness,room_width), texture='WoodFloor.jpg')
s1 = vp.box(pos=vp.vector(room_width/2,wall_height/2,0), size=vp.vector(thickness,wall_height,room_width), texture='Brick.jpg')
s2 = vp.box(pos=vp.vector(-room_width/2,wall_height/2,0), size=vp.vector(thickness,wall_height,room_width), texture='Brick.jpg')
s3 = vp.box(pos=vp.vector(0,wall_height/2,room_width/2), size=vp.vector(room_width,wall_height,thickness), texture='Brick.jpg')
s4 = vp.box(pos=vp.vector(0,wall_height/2,-room_width/2), size=vp.vector(room_width,wall_height,thickness), texture='Brick.jpg')

print(scene.camera.pos)
scene.range = 20

positions = []
prev_angle = 0

'''
with open('positions.pickle', 'rb') as f:
    positions = pickle.load(f)
    print(len(positions[0]))
    for i, p in enumerate(positions[1]):
        cam_x = p[0] 
        cam_z = p[1] 
        cam_y = cam_height
        angle = p[2]

        print(i)
        scene.camera.pos = vp.vector(cam_x, cam_y, cam_z)
        scene.camera.rotate(angle=angle, axis=vp.vector(0,1,0))
        time.sleep(5)
        scene.camera.rotate(angle=-angle, axis=vp.vector(0,1,0))
time.sleep(50)
'''

# Start the loop
for i in range(start_num, start_num + data_points):
    num_objects = r.randint(1, max_obs)
     
    objects = r.sample(all_objects, num_objects)
    for o in objects:
        # First select a random color
        c = r.choice(color_options)
        o.color = c
        
        # Pick coordinates within the room
        x = r.uniform(-room_width/2.0*room_frac, room_width/2.0*room_frac)
        z = r.uniform(-room_width/2.0*room_frac, room_width/2.0*room_frac)
        
        # Randomize the size
        if (type(o) == vp.box):
            o.size = vp.vector(min_width + r.random()*(max_width-min_width), min_height + r.random()*(max_height - min_height), max_width + r.random()*(max_width - min_width))
            o.pos = vp.vector(x, o.size.y/2.0, z)
        elif (type(o) == vp.cone):
            o.axis = vp.vector(0, min_height + r.random()*(max_height - min_height), 0)
            o.radius = max_width/2.0 + r.random()*(max_width - min_width)/2.0
            o.pos = vp.vector(x, 0, z)
        elif (type(o) == vp.sphere):
            o.radius = min_width/2.0 + r.random() * (max_width - min_width)/2.0
            o.pos = vp.vector(x, o.radius, z)
        
        # Make the object visible
        o.visible = True
    
    # Take screen catpures
    positions.append([])
    for j in range(samples_per_scene):
        # Set the camera to a random position
        cam_x = r.uniform(-room_width/2.0*0.9, room_width/2.0*0.9)
        cam_z = r.uniform(-room_width/2.0*0.9, room_width/2.0*0.9)
        cam_y = cam_height
        
        scene.camera.pos = vp.vector(cam_x, cam_y, cam_z)
        # Focus it on a random center
        # cen_x = r.uniform(-room_width/2.0*0.2, room_width/2.0*0.2)
        # cen_z = r.uniform(-room_width/2.0*0.2, room_width/2.0*0.2)
        # cen_y = cam_height
        
        # scene.center = vp.vector(cen_x, cen_y, cen_z)
        # scene.center = vp.vector(0,0,0)
        angle = r.uniform(0,2*np.pi)
        scene.camera.rotate(angle=angle, axis=vp.vector(0,1,0))
        
        new_angle = (prev_angle + angle) % (2*np.pi)
        positions[-1].append([cam_x, cam_z, new_angle])
        prev_angle = new_angle

        # print(vp.mag(scene.center - scene.camera.pos))
        # print(scene.center)
        # print(scene.camera.pos)
        scene.capture('scene_' + str(i) + '_' + str(j) + '.png')
        
        # Gives time for the capture to happen
        time.sleep(1.0) 

    # Make the objects invisible again
    for o in objects:
        o.visible = False
    time.sleep(1.5)

with open('positions.pickle', 'wb') as f:
    pickle.dump(positions, f)
print('Done')

import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from wb_env import whiteboard_wipping
from robot import UR
from utilities import Camera
import time
import math

camera = Camera((0, 2.5, 0.8),
                    (0, 0, 0.8),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)

robot = UR((0.4, 0.4, 0), (0, 0, 0))
env = whiteboard_wipping(robot, camera, vis = True)

env.reset()

"""
action_list = [[x1, y1, z1], [x2, y2, z2]]
"""
# while True:
     # obs, reward, done = env.step((-0.3, -0.5, 0.3, 1.570796251296997, 1.570796251296997, 1.570796251296997), 'end')
obs, reward, done = env.step([-0, -0.2, 0.3], 'end')
env.uptown_funk(120)
file = open('helpme.txt', 'w')
file.write(str(obs))
file.write(str(reward))
file.close()
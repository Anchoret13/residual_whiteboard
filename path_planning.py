import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
from robot import UR, UR5Robotiq85
from utilities import Camera

from config import *

import pybullet_object_models
from pybullet_object_models import ycb_objects

 
from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
import os
import numpy as np

import gym
from gym import spaces

class MOVE_env(gym.core.Env):
    SIMULATION_STEP_DELAY = 1 / 30

    def __init(self, robot, camera, ycb_name, ycb_pos, ycb_ori, vis) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)

        self.camera = camera
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        self.planeID = p.loadURDF("plane.urdf")
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2
        self.gripper_opening_length_control = 0.04

        self.base_obs = []
        self.goal_dim = 3
        self.state_dim = len(self.base_obs)
        self.obs_low = np.array([-1] * self.state_dim)
        self.obs_high = np.array([1] * self.state_dim)
        self.observation_space = spaces.Dict(dict([
            # TODO: desired goal, achieved goal, observation
        ]))

        self.action_space = spaces.Box(-1, 1, shape = (7,),dtype = 'float32')
        self.ycb_name = 'YcbHammer'
        self.path_to_urdf = os.path.join(ycb_objects.getDataPath(), self.ycb_name, "model.urdf")
        
        self.ycb_pos = ycb_pos
        self.ycb_ori = ycb_ori
        self.ycb = p.loadURDF(self.path_to_urdf, self.ycb_pos, self.ycb_ori)

        self.goal = self.sample_goal()
        self.reward_type = 'sparse'
        self.integration_step = 0.01
        p.setTimeStep(self.integration_step)
        self.distance_threshold = 0.1
        self.saved_state = p.saveState()
        p.restoreState(self.saved_state)

        self.reset()

    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def uptown_funk(self, time = 0.5):
        # STOP! WAIT A MINUTE
        steps = int(time/self.integration_step)
        for _ in range(steps):  
            self.step_simulation()
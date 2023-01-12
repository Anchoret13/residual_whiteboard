import time
import math
import random
import numpy as np
import pybullet as p
import pybullet_data
 
from utilities import Models, Camera
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
import os
import numpy as np

import gym
from gym import spaces

module_path = os.path.dirname(__file__)


class whiteboard_wipping(gym.core.Env):
    SIMULATION_STEP_DELAY = 1 / 240

    def __init__(self, robot, camera = None, vis = False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        
        self.camera = camera

        #define env
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # position
        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2

        self.state_dim = 3

        self.board_pos = [0, -0.5, 0.5]
        self.board_ori = p.getQuaternionFromEuler([0, np.pi/2, np.pi/2])
        # plane URDF
        self.planeID = p.loadURDF("./urdf/objects/table.urdf",
                                  self.board_pos, self.board_ori,
                                  useFixedBase = True,
                                  flags = p.URDF_USE_SELF_COLLISION)

        # self.base_obs = np.array([])
        # self.obs_dim = len(self.base_obs)
        # self.obs_low = np.array([-1] * self.state_dim)
        # self.obs_high = np.array([1] * self.state_dim)
        # self.observation_space = -1 # TODO
        self.action_space = spaces.Box(-1 ,1, shape = (9,), dtype = 'float32')
        
        self.distance_upper_limit = 0.03
        self.distance_lower_limit = 0.02

        # 100000 points
        surface_x = np.linspace(self.board_pos[0]-0.8, self.board_pos[0]+0.8, 100)
        surface_z = np.linspace(self.board_pos[2]-0.4, self.board_pos[2]+0.4, 100)

        self.visited_map = []
        
        for i in surface_x:
            for j in surface_z:
                self.visited_map.append([i,j,False])

        self.reward_type = 'sparse'

    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def uptown_funk(self, time = 120):
        # STOP! WAIT A MINUTE
        for _ in range(time):  
            self.step_simulation()

    def step(self, action, control_method = 'end'):
        """
        action: (x, y, z, r, p, y)
        fixed_action: (x, y, z)
        """
        xyz = action.copy()
        previous_visited = self.visited_map.copy()
        
        action.append(0)                # r
        action.append(np.pi)            # p
        action.append(np.pi/2)          # y
        self.robot.move_ee(action, control_method)
        self.uptown_funk()
        self.update_visited(xyz)
        reward = self.update_reward(previous_visited, self.visited_map)
        done = True if reward == 1 else False
        obs = self.get_observation()
        return obs, reward, done

    def is_success(self, xyz):
        board_pos = self.board_pos
        size = [0.8, 1.6, 0.05]
        if board_pos[1] < 0:
            surface_y = board_pos[1]+size[2]/2
        x_range = [board_pos[0]-0.8, board_pos[0]+0.8]
        z_range = [board_pos[2]-0.4, board_pos[2]+0.4]
        distance = np.abs(xyz[1] - surface_y)
        if distance > self.distance_lower_limit and distance < self.distance_upper_limit and xyz[1]>surface_y:
            return True
        return False

    def update_visited(self, xyz):
        board_pos = self.board_pos
        size = [0.8, 1.6, 0.05]
        x_range = 1.6/100
        z_range = 0.8/100
        for i in range(len(self.visited_map)):
            if np.abs(self.visited_map[i][0] - xyz[0]) < x_range and np.abs(self.visited_map[i][1] - xyz[1]) < z_range:
                self.visited_map[i][2] = True

    def update_reward(self, previous_visited, current_visited):
        previous_count = 0
        for item in previous_visited:
            if item[2] == True:
                previous_count += 1
        current_count = 0
        for item in current_visited:
            if item[2] == True:
                current_count += 1
        reward = current_count - previous_count

    def get_observation(self):
        # BASED ON CURRENT URDF
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb = rgb, depth = depth, seg = seg))
        else:
            assert self.camera is None
        obs.update(dict(visited = self.visited_map))
        obs.update(self.robot.get_joint_obs())
        

    def get_observation_cam(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb = rgb, depth = depth, seg = seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset(self):
        self.robot.reset()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)
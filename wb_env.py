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
        self.observation_space = spaces.Dict(dict(
            rgb      = spaces.Box(0, 255, shape = (320, 320, 4)),
            depth    = spaces.Box(0, 255, shape = (320, 320)),
            seg      = spaces.Box(0, 255, shape = (320, 320)),
            visited  = spaces.Box(0, 100, shape = (10000, 3)),
        ))
        self.action_space = spaces.Box(-1 ,1, shape = (3,), dtype = 'float32')
        
        self.distance_upper_limit = 0.02
        self.distance_lower_limit = 0.01

        # 100000 points
        surface_x = np.linspace(self.board_pos[0]-0.8, self.board_pos[0]+0.8, 100)
        surface_z = np.linspace(self.board_pos[2]-0.4, self.board_pos[2]+0.4, 100)

        self.visited_map = []
        
        for i in surface_x:
            for j in surface_z:
                self.visited_map.append([i,j,0])

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
        previous_visited = self.visited_map.copy()
        
        # action.append(0)                # r
        action = np.append(action, 0)
        # action.append(np.pi)            # p
        action = np.append(action, np.pi)
        # action.append(np.pi/2)          # y
        action = np.append(action, np.pi/2)
        self.robot.move_ee(action, control_method)
        xyz, _ = p.getBasePositionAndOrientation(self.robot.id)
        self.uptown_funk()
        self.update_visited(xyz)
        reward = self.update_reward(previous_visited, self.visited_map)
        done = True if reward == 1 else False 
        obs = self.get_observation()
        info = {"episode": 1}
        return obs, reward, done, info

    def is_success(self, xyz):
        # ENV fixed
        size = [0.8, 1.6, 0.05]
        surface_y = self.board_pos[1]+size[2]/2
        distance = np.abs(xyz[1] - surface_y)
        if distance > self.distance_lower_limit and distance < self.distance_upper_limit and xyz[1]>surface_y:
            return 1
        return 0

    def update_visited(self, xyz):
        x_range = 1.6/100
        z_range = 0.8/100
        for i in range(len(self.visited_map)):
            if np.abs(self.visited_map[i][0] - xyz[0]) < x_range and np.abs(self.visited_map[i][1] - xyz[2]) < z_range and self.is_success(xyz):
                self.visited_map[i][2] = True

    def update_reward(self, previous_visited, current_visited):
        previous_count = 0
        for item in previous_visited:
            if item[2] == 1:
                previous_count += 1
        current_count = 0
        for item in current_visited:
            if item[2] == 1:
                current_count += 1
        reward = current_count - previous_count

        return reward

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
        
        return obs

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


def make_wipping_env():
    import os
    from wb_env import whiteboard_wipping
    from robot import UR
    from utilities import Camera
    camera = Camera((0, 2.5, 0.8),
                    (0, 0, 0.8),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)

    robot = UR((0.5, 0.1, 0), (0, 0, 0))
    return whiteboard_wipping(robot, camera, vis = False)

class whiteboard_wipping_diff(whiteboard_wipping):
    def __init__(self, robot, camera=None, vis=False):
        # action space: r, p, y
        self.action_space = spaces.Box(-1 ,1, shape = (4,), dtype = 'float32')

    def step(self, base_action, diff_action, control_method = "end"):
        
        previous_visited = self.visited_map.copy()

        action = np.append(base_action, diff_action)
        self.robot.move_ee(action, control_method)
        xyz, _ = p.getBasePositionAndOrientation(self.robot.id)
        self.uptown_funk()
        self.update_visited(xyz)
        reward = self.update_reward(previous_visited, self.visited_map)
        done = True if reward == 1 else False 
        obs = self.get_observation()
        info = {"episode": 1}
        return obs, reward, done, info

    def update_reward(self, previous_visited, current_visited):
        pass
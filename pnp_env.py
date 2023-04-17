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


class PNP_env(gym.core.Env):
    SIMULATION_STEP_DELAY = 1 / 30 # image 

    def __init__(self, robot, camera, ycb_name, ycb_pos, ycb_ori, vis) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera
        # self.camera_2 = camera_2

        #define env
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation


        # ROBOT POSITION
        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2
        self.gripper_opening_length_control = 0.04

        self.object_grasped = False
        
        # state space
        self.base_obs = [] # TO BE UPDATE
        self.goal_dim = 3 # TODO: update state dim
        self.state_dim = len(self.base_obs)
        self.obs_low = np.array([-1] * self.state_dim)
        self.obs_high = np.array([1] * self.state_dim)
        self.observation_space = spaces.Dict(dict([
            # TODO: desired goal, achieved goal, observation
        ]))

        self.action_space = spaces.Box(-1, 1, shape = (7,),dtype = 'float32')

        # INIT YCB
        # self.ycb_name = ycb_name
        self.ycb_name = 'YcbHammer'
        # ycb_name: {'YcbBanana', 'YcbChipsCan', 'GelatinBox', 'YcbHammer', 'YcbTennisBall'}
        self.path_to_urdf = os.path.join(ycb_objects.getDataPath(), self.ycb_name, "model.urdf")
        
        self.ycb_pos = ycb_pos
        self.ycb_ori = ycb_ori

        self.ycb = p.loadURDF(self.path_to_urdf, self.ycb_pos, self.ycb_ori)
        
        # self.action_space = spaces.Box(-1, 1, shape = (7, ) dtype = 'float32')
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

    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return d < self.distance_threshold

    def grasping_success(self):
        pass

    def drop_penalty(self):
        pass

    def picking_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d<self.distance_threshold)

    def update_reward(self):
        pass

    def goal_distance(goal_a, goal_b):
        goal_a = np.array(goal_a)
        goal_b = np.array(goal_b)
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def step(self, action, control_method = "end"):
        """
        move: (x, y, z, r, p, y, open_length)
        """
        assert control_method in ('end', 'joint')
        grip = (action[6]+1)/2*0.085 # normalization
        base_pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        absolute_pos = action[:6]
        absolute_pos[:3] += np.array(base_pos)
        # absolute_pos[3:6] = (absolute_pos[3:6] + 1/2)*np.pi
        absolute_pos[3] = (absolute_pos[3] + 1/2)*np.pi
        absolute_pos[4] = (absolute_pos[4] + 1/2)*np.pi
        absolute_pos[5] = (absolute_pos[5] + 1/2)*np.pi

        # self.robot.move_ee(absolute_pos, control_method)
        self.robot.move_ee(absolute_pos, control_method)
        self.robot.move_gripper(action[-1])
        delta_t = 0.2
        
        ## PICK

        ## PLACE

        # state = self.get_state()
        state = self.get_observation()
        reward = self.update_reward()
        done = True if reward == 0 else False
        info = {"is_success": done}
        return state, reward, done, info

    def get_state(self):
        pass

    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def get_ycb_obs(self):
        pos, ori = p.getBasePositionAndOrientation(self.ycb)
        return pos, ori
    
    def get_robot_obs(self):
        # joint position: 12
        # joint velocity: 12
        # ee position: 3
        return(self.robot.get_joint_obs())
    
    def sample_goal(self):
        pass

    def reset_env(self):
        p.resetBasePositionAndOrientation(self.ycb, self.ycb_pos, self.ycb_ori)
        self.robot.move_ee((0, 0, 0.5, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')

    def reset(self):
        self.goal = self.sample_goal()
        self.goal_vis = 0 # VISUALIZE WITH A NON-COLLISION GOAL
        self.robot.reset()
        self.reset_env()
        self.object_grasped = False
        state = self.get_observation()
        return state

    def close(self):
        p.disconnect(self.physicsClient)
    

def make_PNP():
     camera_1 = Camera((0, 2.5, 0.8),
                         (0, 0, 0.8),
                         (0, 0, 1),
                         0.1, 5, (320, 320), 40)
     camera_2 = Camera((0, 0, 2.5),
                         (0, 0, 0.8),
                         (0, 0, 1),
                         0.1, 5, (320, 320), 40)
     robot = UR5Robotiq85((0.5, 0.1, 0), (0, 0, 0))
     ycb = 0
     ycb_pos = [0, -0.5, 0.1]
     ycb_ori = p.getQuaternionFromEuler([0, np.pi/2, np.pi/2])
     
     env = PNP_env(robot, camera=camera_1, ycb_name= ycb, ycb_pos = ycb_pos, ycb_ori = ycb_ori,vis = True)
     return env
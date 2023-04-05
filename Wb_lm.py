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

class wb_lm(gym.core.Env):
    SIMULATION_STEP_DELAY = 1/100

    def __init__(self, robot, camera = None, vis = False):
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

        # move speed of each time step
        self.velocity = 0.05

        # position
        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2

        self.state_dim = 3

        self.board_pos = [0, -0.35, 0.5]
        self.board_ori = p.getQuaternionFromEuler([0, np.pi/2, np.pi/2])
        # plane URDF
        self.planeID = p.loadURDF("./urdf/objects/table.urdf",
                                  self.board_pos, self.board_ori,
                                  useFixedBase = True,
                                  flags = p.URDF_USE_SELF_COLLISION)

        self.observation_space = spaces.Dict(dict(
            rgb      = spaces.Box(0, 255, shape = (320, 320, 4)),
            depth    = spaces.Box(0, 255, shape = (320, 320)),
            seg      = spaces.Box(0, 255, shape = (320, 320))
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

    def lm_center_size(self, pos_dirt, height, width):
        control_method = 'end'
        x = pos_dirt[0]
        fixed_y = pos_dirt[1]
        z = pos_dirt[2]
        init_x = x - width/2 - 3 * self.velocity
        init_z = z - height/2 - 1 * self.velocity

        distance = 0.2
        init_y = -0.5 + distance

        stop_x = x + width/2 + 3 * self.velocity
        stop_z = z - height/2 - 1 * self.velocity

        vertical = height/(2*self.velocity)
        horizontal = width/self.velocity
        i = 0
        j = 0
        current_x = init_x
        current_z = init_z
        while current_z < stop_z:
            while current_x < stop_x:
                current_x = current_x + self.velocity * self.SIMULATION_STEP_DELAY
                robot_action = [current_x, fixed_y, current_z, 0, np.pi/2, np.pi/2]
                self.robot.move_ee(robot_action, control_method)
                self.uptown_funk()
            if current_x > stop_x:
                current_z += self.velocity * 2 * self.SIMULATION_STEP_DELAY
                robot_action = [current_x, fixed_y, current_z, 0, np.pi/2, np.pi/2]
                self.robot.move_ee(robot_action, control_method)
                self.uptown_funk()
            while current_x > init_x:
                current_x = current_x - self.velocity * self.SIMULATION_STEP_DELAY
                robot_action = [current_x, fixed_y, current_z, 0, np.pi/2, np.pi/2]
                self.robot.move_ee(robot_action, control_method)
                self.uptown_funk()
            if current_x < init_x:
                current_z += self.velocity * 2 * self.SIMULATION_STEP_DELAY
                robot_action = [current_x, fixed_y, current_z, 0, np.pi/2, np.pi/2]
                self.robot.move_ee(robot_action, control_method)
                self.uptown_funk()

    def reset(self):
        self.robot.move_ee((0, -0.15, 0.75, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')

    def step(self, action, lm_tr = 'lm'):
        """
        'or': [pos, ori]                   # pos: [x, y, z], ori: [r, p, y]
        'lm': [pos_dirt, height, width]    # pos_dirt: [x, y, z] height, width
        """
        if lm_tr == 'lm':
            pos_dirt = [action[0], action[1], action[2]]
            height = action[3]
            width = action[4]
            # self.lm_center_size(pos_dirt, height, width)

        if lm_tr == 'or':
            # base_pos, _ = p.getBasePositionAndOrientation(self.robot.id)
            base_pos = [0.3, -0.3, 0.5]
            base_ori = [0, np.pi/2, np.pi/2]
            absolute_pos = action[:6]
    
def main():
    from robot import UR
    from utilities import Camera
    camera = Camera((4, 0, 1),
                    (0, -0.7, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    
    robot = UR((0.5, 0.1, 0), (0, 0, 0))
    
    env = wb_lm(robot, camera, vis = True)
    count = 0

    # sample_action = [0.3,0.3,0.3,0.5,0.2]
    sample_action = [0.01, 0.01, 0.01, 0, np.pi/2, np.pi/2]
    # while count < 10000:
    #     env.step(sample_action, lm_tr="or")
    # env.robot.move_ee([-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
    #                            -1.5707970583733368, 0.0009377758247187636], 'end')
    def wipe_1(z):
        env.robot.move_ee([0.3, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.3, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.28, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.26, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.24, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.22, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.20, -0.3, z, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.18, -0.3, z-0.01, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.18, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.20, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.22, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.24, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.26, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
        env.robot.move_ee([0.28, -0.3, z-0.02, 0, 0, -np.pi/2], 'end')
        env.uptown_funk(20)
    env.robot.move_ee([0.3, -0.3, 0.4, 0, 0, -np.pi/2], 'end')
    env.uptown_funk(4000)
    wipe_1(0.4)
    wipe_1(0.36)
    wipe_1(0.32)
    wipe_1(0.28)
    wipe_1(0.24)
        



    env.uptown_funk(24000)

main()
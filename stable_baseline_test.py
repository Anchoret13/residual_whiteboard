import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from wb_env import whiteboard_wipping, make_wipping_env
from robot import UR
from utilities import Camera
import time
import math

from stable_baselines3 import PPO

# def origin_sb3_test():
#     from stable_baselines3.common.env_util import make_vec_env
#     env = make_vec_env("CartPole-v1", n_envs=4)
#     model = PPO("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=2)
#     model.save("ppo_cartpole")

#     del model # remove to demonstrate saving and loading

#     model = PPO.load("ppo_cartpole")

#     obs = env.reset()
#     while True:
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         env.render()

#     print(info)

if __name__ == "__main__":
    env = make_wipping_env()
    env.reset()
    model = PPO('MultiInputPolicy', env, verbose = 1)
    model.learn(total_timesteps= 1000)
    model.save('test')

    # origin_sb3_test()
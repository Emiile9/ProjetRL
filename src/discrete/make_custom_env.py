import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import numpy as np

def make_env(continuous: bool):
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=continuous)
    
    env = GrayscaleObservation(env, keep_dim=False)

    env = ResizeObservation(env, shape=(64, 64))

    env = FrameStackObservation(env, stack_size=4)

    return env

env = make_env(False)
obs, _ = env.reset()
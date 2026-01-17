import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
import numpy as np

def make_env(continuous : bool):

    env = gym.make("CarRacing-v3", render_mode = "rgb_array", continuous = continuous)

    grey_scaled_env = GrayscaleObservation(env, keep_dim=True)

    resized_env = ResizeObservation(grey_scaled_env, shape=(64, 64))

    stacked_env = FrameStackObservation(resized_env, 4)

    return stacked_env

env = make_env(False)
obs, _ = env.reset()
print(type(env))
print(obs.shape)

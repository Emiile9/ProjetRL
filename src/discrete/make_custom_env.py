import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, MaxAndSkipObservation
import numpy as np

def make_env(continuous: bool, render_mode = "rgb_array", mode = "train"):
    if mode == "watch":
        render_mode = "human"
        max_episode_steps = 2000
    else:
        max_episode_steps = 1000
    env = gym.make("CarRacing-v3", render_mode=render_mode, continuous=continuous, max_episode_steps=max_episode_steps)

    env = MaxAndSkipObservation(env, skip=4)

    env = GrayscaleObservation(env, keep_dim=False)

    env = ResizeObservation(env, shape=(64, 64))

    env = FrameStackObservation(env, stack_size=4)

    return env

env = make_env(False)
obs, _ = env.reset()
import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

import gymnasium as gym

# Use "human" to see the window, or "rgb_array" for training
env = gym.make("CarRacing-v3", render_mode="human")

observation, info = env.reset()

for _ in range(10000):
    # Action space: [steering, gas, braking] 
    # e.g., [0.5, 0.1, 0.0] is turning right while accelerating slightly
    action = env.action_space.sample() 
    
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
from src.utils.make_custom_env import make_env
from src.dqn.dqn_discrete import DQNDiscrete
import torch
import numpy as np


def watch_agent(path, episodes=100):
    env = make_env(render_mode="human", continuous=False, mode="watch")
    agent = DQNDiscrete(action_space=5)
    agent.load(path)
    agent.policy_network.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            device = agent.device
            with torch.no_grad():
                state_tensor = (
                    torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
                )
                action = agent.policy_network(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(
            f"Episode {ep} finished. Score: {total_reward:.2f} | Terminated: {terminated} | Truncated: {truncated}"
        )
    env.close()


watch_agent("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/car_dqn_0_05_1400.pth")

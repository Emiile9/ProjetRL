import torch
import numpy as np
import time
from src.utils.make_custom_env import make_env
from src.ppo.ppo import PPO

from src.ppo.actor_critic import ActorCritic  # Import the model class directly


def watch_ppo_agent(path, episodes=5):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    env = make_env(render_mode="human", continuous=True)

    # 1. Initialize only the Model (the "Brain")
    model = ActorCritic().to(device)

    # 2. Load the weights
    # Using weights_only=True is a good security practice in newer Torch versions
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Pre-process observation
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                mu, _ = model(obs_t)
                action = mu.cpu().numpy()[0]

            env_action = np.array(
                [
                    action[0],  # Steer
                    (action[1] + 1.0) / 2.0,  # Gas
                    (action[2] + 1.0) / 2.0,  # Brake
                ]
            )

            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1} Finished | Score: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # Ensure the path points to your saved .pth file
    watch_ppo_agent("/Users/emile/Desktop/IASD/TP_M2/ProjetRL/ppo_carracing_ep5600.pth")

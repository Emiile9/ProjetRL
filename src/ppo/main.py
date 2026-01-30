import gymnasium as gym
from src.ppo.actor_critic import ActorCritic
from src.ppo.ppo import PPO
from src.utils.make_custom_env import make_env
import torch

def train_agent(env, total_steps=1_000_000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_critic = ActorCritic().to(device)
    ppo = PPO(actor_critic, trajectories=2048)
    ppo.device = device  # Ensure PPO has access to the device
    # Implementation of the master loop
    current_steps = 0
    while current_steps < total_steps:
        # Step 1: Collect
        obs, acts, rews, dones, lps, vals, last_val = ppo.collect_trajectories(env)
        
        # Step 2: GAE
        adv, returns = ppo.compute_advantages(rews, vals, dones, last_val)

        # Step 3: Optimize
        ppo.update(obs, acts, lps, returns, adv)

        current_steps += ppo.trajectories
        print(f"Total Steps: {current_steps} | Avg Reward: {rews.sum().item():.2f}")

        if current_steps % (ppo.trajectories * 10) == 0:
            ppo.save(f"actor_critic_weights_{current_steps}.pth")
            print("Checkpoint saved.")

    env.close()

env = make_env(continuous=True)
train_agent(env)
import gymnasium as gym
from src.ppo.actor_critic import ActorCritic
from src.ppo.ppo import PPO
from src.utils.make_custom_env import make_env
import torch
from torch.distributions import Normal
import pickle


import gymnasium as gym
import numpy as np

def main(lr=1e-4, epochs=4, batch_size=128, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, gae_lambda=0.95, gamma=0.99):
    CONTINUOUS = True
    TOTAL_TIMESTEPS = 2_000_000 
    ROLLOUT_LENGTH = 2048
    
    # Créer l'environnement
    env = make_env(continuous=CONTINUOUS, mode="train")
    

    agent = PPO(
        env=env,
        lr=lr,              
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_ratio=0.2,
        epochs=epochs,           
        batch_size=batch_size,       
        value_coef=value_coef,
        entropy_coef=entropy_coef,   
        max_grad_norm=max_grad_norm
    )
    
    print("Début de l'entraînement OPTIMISÉ...")
    print(f"Type d'action: {'Continu' if CONTINUOUS else 'Discret'}")
    print(f"Device: {agent.device}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    
    # Entraîner l'agent
    rewards = agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        rollout_length=ROLLOUT_LENGTH,
        log_interval=10,
        save_interval=100
    )
    
    # Sauvegarder le modèle final
    agent.save("ppo_carracing_final.pth")
    
    
    print("Entraînement terminé!")
    print(f"Meilleure récompense moyenne: {max([np.mean(rewards[i:i+10]) for i in range(len(rewards)-10)]):.2f}")
    with open("stats_ppo.pkl", "wb") as f:
        pickle.dump(rewards, f)

    return rewards

if __name__ == "__main__":
    stats = main()
    with open("stats_ppo.pkl", "wb") as f:
        pickle.dump(stats, f)
